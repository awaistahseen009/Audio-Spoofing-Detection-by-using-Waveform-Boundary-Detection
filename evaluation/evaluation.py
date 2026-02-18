import os
import sys
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np
import torch
from sklearn.metrics import (
    roc_curve, classification_report, confusion_matrix, roc_auc_score
)
from src.inference import load_model, infer_single_audio
from utils.logger import setup_logger

logger = setup_logger(__name__, log_type='evaluation')

# -----------------------------------------
# CONFIG
# -----------------------------------------
CHECKPOINT_PATH = "checkpoints/checkpoint_epoch_4.pt"
REAL_DIR        = "Real"
FAKE_DIR        = "Fake"
THRESHOLD       = 0.5       # frame-level threshold
SPOOF_THRESHOLD = 50.0      # if fake_percentage > this => file is spoofed
SUPPORTED       = ('.wav', '.mp3', '.flac', '.ogg')


# -----------------------------------------
# EER CALCULATION
# -----------------------------------------
def calculate_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float(fpr[idx]) * 100, float(thresholds[idx])


# -----------------------------------------
# PROCESS ONE FOLDER
# -----------------------------------------
def process_folder(model, folder_path, true_label, device):
    results = []
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(SUPPORTED)]

    if not files:
        logger.warning(f"No audio files found in {folder_path}")
        return results

    label_str = "FAKE" if true_label else "REAL"
    logger.info(f"Processing {len(files)} files from '{folder_path}' (label={label_str})...")

    for fname in sorted(files):
        fpath = os.path.join(folder_path, fname)
        try:
            t0 = time.time()
            result = infer_single_audio(model, fpath, device, threshold=THRESHOLD)
            elapsed = time.time() - t0

            fake_pct  = float(result["fake_percentage"])
            predicted = 1 if fake_pct > SPOOF_THRESHOLD else 0
            correct   = predicted == true_label

            logger.info(
                f"[{label_str}] {fname} | "
                f"Fake frames: {fake_pct:.2f}% ({result['fake_frames']}/{result['total_frames']}) | "
                f"Predicted: {'SPOOF' if predicted else 'REAL'} | "
                f"{'OK' if correct else 'WRONG'} | "
                f"Time: {elapsed:.2f}s"
            )

            results.append({
                "filename":        fname,
                "true_label":      true_label,
                "predicted":       predicted,
                "fake_percentage": fake_pct,
                "total_frames":    result["total_frames"],
                "fake_frames":     result["fake_frames"],
                "correct":         correct,
                "score":           fake_pct / 100.0,
                "elapsed":         elapsed
            })

        except Exception as e:
            logger.error(f"Failed to process {fname}: {e}")

    return results


# -----------------------------------------
# LOG METRICS
# -----------------------------------------
def log_metrics(all_results, model_name="PyTorch Checkpoint"):
    if not all_results:
        logger.error("No results to evaluate.")
        return {}

    labels    = np.array([r["true_label"]      for r in all_results])
    predicted = np.array([r["predicted"]        for r in all_results])
    scores    = np.array([r["score"]            for r in all_results])
    elapsed   = np.array([r["elapsed"]          for r in all_results])

    total    = len(all_results)
    correct  = int(sum(r["correct"] for r in all_results))
    accuracy = correct / total * 100

    eer, eer_thresh = calculate_eer(labels, scores)
    auc = roc_auc_score(labels, scores)

    real_results = [r for r in all_results if r["true_label"] == 0]
    fake_results = [r for r in all_results if r["true_label"] == 1]
    real_avg_pct = np.mean([r["fake_percentage"] for r in real_results]) if real_results else 0.0
    fake_avg_pct = np.mean([r["fake_percentage"] for r in fake_results]) if fake_results else 0.0

    logger.info("=" * 60)
    logger.info(f"RESULTS - {model_name}")
    logger.info("=" * 60)
    logger.info(f"Total files          : {total}  (Real={len(real_results)}, Fake={len(fake_results)})")
    logger.info(f"Accuracy             : {accuracy:.2f}%  ({correct}/{total})")
    logger.info(f"EER                  : {eer:.2f}%  (threshold={eer_thresh:.4f})")
    logger.info(f"AUC-ROC              : {auc:.4f}")
    logger.info(f"Avg inference time   : {np.mean(elapsed):.2f}s per file")
    logger.info("-" * 60)
    logger.info(f"Avg fake frame % - REAL files : {real_avg_pct:.2f}%")
    logger.info(f"Avg fake frame % - FAKE files : {fake_avg_pct:.2f}%")
    logger.info("-" * 60)

    cm = confusion_matrix(labels, predicted)
    logger.info("Confusion Matrix (rows=true, cols=pred):")
    logger.info("               Pred REAL   Pred FAKE")
    if cm.shape == (2, 2):
        logger.info(f"  True REAL  :   {cm[0][0]:>6}      {cm[0][1]:>6}")
        logger.info(f"  True FAKE  :   {cm[1][0]:>6}      {cm[1][1]:>6}")
    logger.info("-" * 60)

    report = classification_report(labels, predicted, target_names=["REAL", "FAKE"], digits=4)
    for line in report.splitlines():
        logger.info(line)

    wrong = [r for r in all_results if not r["correct"]]
    if wrong:
        logger.warning(f"Misclassified files ({len(wrong)}):")
        for r in wrong:
            logger.warning(
                f"  {r['filename']} | "
                f"True: {'FAKE' if r['true_label'] else 'REAL'} | "
                f"Pred: {'FAKE' if r['predicted'] else 'REAL'} | "
                f"Fake%: {r['fake_percentage']:.2f}%"
            )
    else:
        logger.info("All files classified correctly!")

    logger.info("=" * 60)

    return {
        "model":        model_name,
        "accuracy":     accuracy,
        "eer":          eer,
        "auc":          auc,
        "avg_time":     float(np.mean(elapsed)),
        "real_avg_pct": real_avg_pct,
        "fake_avg_pct": fake_avg_pct,
    }


# -----------------------------------------
# MAIN
# -----------------------------------------
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("STARTING PYTORCH EVALUATION")
    logger.info(f"Checkpoint : {CHECKPOINT_PATH}")
    logger.info(f"Real dir   : {REAL_DIR}")
    logger.info(f"Fake dir   : {FAKE_DIR}")
    logger.info(f"Thresholds : frame={THRESHOLD}, spoof={SPOOF_THRESHOLD}%")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = load_model(CHECKPOINT_PATH, device)

    real_results = process_folder(model, REAL_DIR, true_label=0, device=device)
    fake_results = process_folder(model, FAKE_DIR, true_label=1, device=device)

    log_metrics(real_results + fake_results, model_name="PyTorch Checkpoint")
import os
import sys
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np
import torch
import onnxruntime as ort
import soundfile as sf
import torchaudio
from sklearn.metrics import (
    roc_curve, classification_report, confusion_matrix, roc_auc_score
)
from src.inference import load_model, infer_single_audio
from utils.audio_utils import pad_audio
from utils.logger import setup_logger

logger = setup_logger(__name__, log_type='evaluation')

# -----------------------------------------
# CONFIG
# -----------------------------------------
CHECKPOINT_PATH  = "checkpoints/checkpoint_epoch_4.pt"
ONNX_FP32_PATH   = "checkpoints/model_fp32.onnx"
ONNX_INT8_PATH   = "checkpoints/model_int8.onnx"
REAL_DIR         = "Real"
FAKE_DIR         = "Fake"
THRESHOLD        = 0.5      # frame-level
SPOOF_THRESHOLD  = 50.0     # file-level fake%
SUPPORTED        = ('.wav', '.mp3', '.flac', '.ogg')
SAMPLE_RATE      = 16000
TARGET_DURATION  = 6


# -----------------------------------------
# EER
# -----------------------------------------
def calculate_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float(fpr[idx]) * 100, float(thresholds[idx])


# -----------------------------------------
# AUDIO PREPROCESSING (shared)
# -----------------------------------------
def preprocess_audio(audio_path):
    data, sr = sf.read(audio_path, dtype="float32")
    if data.ndim == 1:
        data = data[np.newaxis, :]
    else:
        data = data.T
    waveform = torch.tensor(data)
    waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
    waveform = pad_audio(waveform, SAMPLE_RATE, TARGET_DURATION)
    return waveform


# -----------------------------------------
# PYTORCH INFERENCE
# -----------------------------------------
def infer_pytorch(model, audio_path, device):
    t0 = time.time()
    result = infer_single_audio(model, audio_path, device, threshold=THRESHOLD)
    elapsed = time.time() - t0
    fake_pct = float(result["fake_percentage"])
    return {
        "fake_percentage": fake_pct,
        "total_frames":    result["total_frames"],
        "fake_frames":     result["fake_frames"],
        "predicted":       1 if fake_pct > SPOOF_THRESHOLD else 0,
        "score":           fake_pct / 100.0,
        "elapsed":         elapsed
    }


# -----------------------------------------
# ONNX INFERENCE
# -----------------------------------------
def infer_onnx(session, audio_path):
    t0 = time.time()
    waveform = preprocess_audio(audio_path)
    input_data = waveform.unsqueeze(0).numpy()  # [1, 1, samples]

    output = session.run(None, {"audio": input_data})[0]  # [1, frames, 1]
    output = output.flatten()

    prediction = (output > THRESHOLD).astype(int)
    fake_pct = float(prediction.sum() / len(prediction) * 100)
    elapsed = time.time() - t0

    return {
        "fake_percentage": fake_pct,
        "total_frames":    int(len(prediction)),
        "fake_frames":     int(prediction.sum()),
        "predicted":       1 if fake_pct > SPOOF_THRESHOLD else 0,
        "score":           fake_pct / 100.0,
        "elapsed":         elapsed
    }


# -----------------------------------------
# PROCESS ONE FOLDER WITH ALL MODELS
# -----------------------------------------
def process_folder(folder_path, true_label, pt_model, pt_device, fp32_sess, int8_sess):
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(SUPPORTED)]
    if not files:
        logger.warning(f"No audio files found in {folder_path}")
        return [], [], []

    label_str = "FAKE" if true_label else "REAL"
    logger.info(f"Processing {len(files)} [{label_str}] files from '{folder_path}'...")

    pt_results, fp32_results, int8_results = [], [], []

    for fname in sorted(files):
        fpath = os.path.join(folder_path, fname)
        try:
            # --- PyTorch ---
            pt   = infer_pytorch(pt_model, fpath, pt_device)
            # --- FP32 ONNX ---
            fp32 = infer_onnx(fp32_sess, fpath)
            # --- INT8 ONNX ---
            int8 = infer_onnx(int8_sess, fpath)

            for tag, res, bucket in [
                ("PT",   pt,   pt_results),
                ("FP32", fp32, fp32_results),
                ("INT8", int8, int8_results),
            ]:
                correct = res["predicted"] == true_label
                res.update({
                    "filename":   fname,
                    "true_label": true_label,
                    "correct":    correct,
                })
                bucket.append(res)

            logger.info(
                f"[{label_str}] {fname} | "
                f"PT: {pt['fake_percentage']:.1f}% ({'OK' if pt['predicted']==true_label else 'WRONG'}) | "
                f"FP32: {fp32['fake_percentage']:.1f}% ({'OK' if fp32['predicted']==true_label else 'WRONG'}) | "
                f"INT8: {int8['fake_percentage']:.1f}% ({'OK' if int8['predicted']==true_label else 'WRONG'}) | "
                f"Time PT={pt['elapsed']:.2f}s FP32={fp32['elapsed']:.2f}s INT8={int8['elapsed']:.2f}s"
            )

        except Exception as e:
            logger.error(f"Failed on {fname}: {e}")

    return pt_results, fp32_results, int8_results


# -----------------------------------------
# LOG METRICS FOR ONE MODEL
# -----------------------------------------
def log_metrics(all_results, model_name):
    if not all_results:
        logger.error(f"No results for {model_name}")
        return {}

    labels    = np.array([r["true_label"] for r in all_results])
    predicted = np.array([r["predicted"]  for r in all_results])
    scores    = np.array([r["score"]      for r in all_results])
    elapsed   = np.array([r["elapsed"]    for r in all_results])

    total    = len(all_results)
    correct  = int(sum(r["correct"] for r in all_results))
    accuracy = correct / total * 100
    eer, eer_thresh = calculate_eer(labels, scores)
    auc = roc_auc_score(labels, scores)

    real_res = [r for r in all_results if r["true_label"] == 0]
    fake_res = [r for r in all_results if r["true_label"] == 1]
    real_avg = np.mean([r["fake_percentage"] for r in real_res]) if real_res else 0.0
    fake_avg = np.mean([r["fake_percentage"] for r in fake_res]) if fake_res else 0.0

    logger.info("=" * 60)
    logger.info(f"RESULTS - {model_name}")
    logger.info("=" * 60)
    logger.info(f"Total files          : {total}  (Real={len(real_res)}, Fake={len(fake_res)})")
    logger.info(f"Accuracy             : {accuracy:.2f}%  ({correct}/{total})")
    logger.info(f"EER                  : {eer:.2f}%  (threshold={eer_thresh:.4f})")
    logger.info(f"AUC-ROC              : {auc:.4f}")
    logger.info(f"Avg inference time   : {np.mean(elapsed):.3f}s per file")
    logger.info("-" * 60)
    logger.info(f"Avg fake frame % - REAL files : {real_avg:.2f}%")
    logger.info(f"Avg fake frame % - FAKE files : {fake_avg:.2f}%")
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
        logger.warning(f"Misclassified ({len(wrong)}):")
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
        "model":    model_name,
        "accuracy": accuracy,
        "eer":      eer,
        "auc":      auc,
        "avg_time": float(np.mean(elapsed)),
    }


# -----------------------------------------
# COMPARISON TABLE
# -----------------------------------------
def log_comparison(summaries):
    logger.info("=" * 60)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Model':<20} {'Accuracy':>10} {'EER':>8} {'AUC':>8} {'Avg Time':>10}")
    logger.info("-" * 60)
    for s in summaries:
        logger.info(
            f"{s['model']:<20} "
            f"{s['accuracy']:>9.2f}% "
            f"{s['eer']:>7.2f}% "
            f"{s['auc']:>8.4f} "
            f"{s['avg_time']:>9.3f}s"
        )
    logger.info("=" * 60)

    # Speed comparison vs PyTorch
    pt = next((s for s in summaries if "PyTorch" in s["model"]), None)
    if pt:
        for s in summaries:
            if s["model"] != pt["model"]:
                speedup = pt["avg_time"] / s["avg_time"] if s["avg_time"] > 0 else 0
                acc_diff = s["accuracy"] - pt["accuracy"]
                eer_diff = s["eer"] - pt["eer"]
                logger.info(
                    f"{s['model']} vs PyTorch | "
                    f"Speedup: {speedup:.2f}x | "
                    f"Accuracy diff: {acc_diff:+.2f}% | "
                    f"EER diff: {eer_diff:+.2f}%"
                )
    logger.info("=" * 60)


# -----------------------------------------
# MAIN
# -----------------------------------------
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("STARTING ONNX vs PYTORCH EVALUATION")
    logger.info(f"Checkpoint  : {CHECKPOINT_PATH}")
    logger.info(f"FP32 ONNX   : {ONNX_FP32_PATH}")
    logger.info(f"INT8 ONNX   : {ONNX_INT8_PATH}")
    logger.info(f"Real dir    : {REAL_DIR}")
    logger.info(f"Fake dir    : {FAKE_DIR}")
    logger.info(f"Thresholds  : frame={THRESHOLD}, spoof={SPOOF_THRESHOLD}%")
    logger.info("=" * 60)

    # Load PyTorch model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    pt_model = load_model(CHECKPOINT_PATH, device)
    logger.info("PyTorch model loaded")

    # Load ONNX sessions
    logger.info("Loading FP32 ONNX session...")
    fp32_sess = ort.InferenceSession(ONNX_FP32_PATH, providers=["CPUExecutionProvider"])
    logger.info("Loading INT8 ONNX session...")
    int8_sess = ort.InferenceSession(ONNX_INT8_PATH, providers=["CPUExecutionProvider"])
    logger.info("All models loaded")

    # Process both folders
    pt_real,   fp32_real,   int8_real   = process_folder(REAL_DIR, 0, pt_model, device, fp32_sess, int8_sess)
    pt_fake,   fp32_fake,   int8_fake   = process_folder(FAKE_DIR, 1, pt_model, device, fp32_sess, int8_sess)

    pt_all   = pt_real   + pt_fake
    fp32_all = fp32_real + fp32_fake
    int8_all = int8_real + int8_fake

    # Log per-model metrics
    pt_summary   = log_metrics(pt_all,   "PyTorch Checkpoint")
    fp32_summary = log_metrics(fp32_all, "FP32 ONNX")
    int8_summary = log_metrics(int8_all, "INT8 ONNX")

    # Final comparison table
    log_comparison([pt_summary, fp32_summary, int8_summary])
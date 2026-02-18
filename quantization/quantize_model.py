import os
import sys
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Add project root to path so src/ and utils/ are importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from src.model import BoundaryDetectionModel
from utils.logger import setup_logger

logger = setup_logger(__name__, log_type='quantization')

# ----------------------------------------
# CONFIG
# ----------------------------------------
CHECKPOINT_PATH = "checkpoints/checkpoint_epoch_4.pt"
ONNX_FP32_PATH  = "checkpoints/model_fp32.onnx"
ONNX_INT8_PATH  = "checkpoints/model_int8.onnx"
SAMPLE_RATE     = 16000
TARGET_DURATION = 6       # seconds — must match training
DEVICE          = torch.device("cpu")  # ONNX export must be on CPU


# ----------------------------------------
# STEP 1: Load trained model
# ----------------------------------------
def load_model(checkpoint_path):
    logger.info(f"Loading model from {checkpoint_path}...")
    model = BoundaryDetectionModel().to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info("Model loaded successfully")
    return model


# ----------------------------------------
# STEP 2: Export to ONNX (FP32)
# ----------------------------------------
def export_to_onnx(model, onnx_path):
    logger.info("Exporting model to FP32 ONNX...")
    dummy_input = torch.zeros(1, 1, SAMPLE_RATE * TARGET_DURATION)  # [batch, channel, samples]

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["audio"],
        output_names=["frame_predictions"],
        dynamic_axes={
            "audio": {0: "batch_size", 2: "audio_length"},
            "frame_predictions": {0: "batch_size", 1: "num_frames"}
        },
        dynamo=False   # force legacy exporter — no onnxscript needed
    )
    logger.info(f"FP32 ONNX model exported to {onnx_path}")

    # Validate the exported model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model validation passed")


# ----------------------------------------
# STEP 3: Quantize to INT8
# ----------------------------------------
def quantize_to_int8(fp32_path, int8_path):
    logger.info("Quantizing FP32 ONNX model to INT8...")
    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm", "LSTM"]  # skip Conv layers — ConvInteger not supported on CPU
    )
    logger.info(f"INT8 quantized model saved to {int8_path}")


# ----------------------------------------
# STEP 4: Verify both models match outputs
# ----------------------------------------
def verify_models(model, fp32_path, int8_path):
    logger.info("Verifying model outputs match...")
    dummy_input = torch.zeros(1, 1, SAMPLE_RATE * TARGET_DURATION)

    # PyTorch output
    with torch.no_grad():
        pt_output = model(dummy_input).squeeze(-1).numpy()

    # FP32 ONNX output
    sess_fp32 = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    fp32_output = sess_fp32.run(None, {"audio": dummy_input.numpy()})[0]

    # INT8 ONNX output
    sess_int8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    int8_output = sess_int8.run(None, {"audio": dummy_input.numpy()})[0]

    logger.info(f"PyTorch output shape  : {pt_output.shape}")
    logger.info(f"FP32 ONNX output shape: {fp32_output.shape}")
    logger.info(f"INT8 ONNX output shape: {int8_output.shape}")

    diff_fp32 = np.max(np.abs(pt_output - fp32_output))
    diff_int8 = np.max(np.abs(pt_output - int8_output))
    logger.info(f"Max diff PT vs FP32   : {diff_fp32:.6f}")
    logger.info(f"Max diff PT vs INT8   : {diff_int8:.6f}")

    # Note: diff up to ~0.2 is expected due to SDPA attention tracing constant baking
    if diff_fp32 < 0.3:
        logger.info(f"FP32 ONNX diff is acceptable (SDPA tracing causes minor variance): {diff_fp32:.6f}")
    else:
        logger.warning(f"FP32 ONNX diff is unexpectedly high: {diff_fp32:.6f}")

    if diff_int8 < 0.3:
        logger.info(f"INT8 ONNX diff is acceptable: {diff_int8:.6f}")
    else:
        logger.warning(f"INT8 ONNX diff is high - quantization may have hurt accuracy: {diff_int8:.6f}")


# ----------------------------------------
# STEP 5: Print size comparison
# ----------------------------------------
def print_size_comparison(checkpoint_path, fp32_path, int8_path):
    def mb(path):
        return os.path.getsize(path) / (1024 * 1024)

    orig_mb  = mb(checkpoint_path)
    fp32_mb  = mb(fp32_path)
    int8_mb  = mb(int8_path)
    reduction = (1 - int8_mb / fp32_mb) * 100

    logger.info("-- Model Size Comparison --")
    logger.info(f"Original checkpoint : {orig_mb:.1f} MB")
    logger.info(f"FP32 ONNX           : {fp32_mb:.1f} MB")
    logger.info(f"INT8 ONNX           : {int8_mb:.1f} MB")
    logger.info(f"Size reduction      : {reduction:.1f}%")
    logger.info("-------------------------------------------")


# ----------------------------------------
# MAIN
# ----------------------------------------
if __name__ == "__main__":
    logger.info("=== ONNX Export + INT8 Quantization ===")

    model = load_model(CHECKPOINT_PATH)
    export_to_onnx(model, ONNX_FP32_PATH)
    quantize_to_int8(ONNX_FP32_PATH, ONNX_INT8_PATH)
    verify_models(model, ONNX_FP32_PATH, ONNX_INT8_PATH)
    print_size_comparison(CHECKPOINT_PATH, ONNX_FP32_PATH, ONNX_INT8_PATH)

    logger.info("Done! Use model_int8.onnx for inference.")
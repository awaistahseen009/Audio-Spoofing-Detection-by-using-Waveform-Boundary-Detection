import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import torch
import tempfile
import shutil
from src.inference import load_model, infer_single_audio
from utils.logger import setup_logger

logger = setup_logger(__name__, log_type='api')

app = FastAPI(
    title="Audio Spoofing Detection API",
    description="Concatenative audio spoofing detection using frame-level boundary detection",
    version="2.0.0"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "checkpoints/checkpoint_epoch_4.pt")

SUPPORTED_FORMATS = ('.wav', '.mp3', '.flac', '.ogg')


@app.on_event("startup")
async def startup_event():
    global model
    logger.info("Loading model...")
    model = load_model(CHECKPOINT_PATH, device)
    logger.info("Model loaded successfully")


@app.get("/")
async def root():
    return {
        "message": "Audio Spoofing Detection API",
        "status": "running",
        "device": str(device)
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


async def process_single_file(file: UploadFile) -> dict:
    """Helper to process one file and return a result dict."""
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name

        logger.info(f"Processing uploaded file: {file.filename}")
        result = infer_single_audio(model, temp_path, device)

        return {
            "filename": file.filename,
            "fake_percentage": float(result["fake_percentage"]),
            "total_frames": int(result["total_frames"]),
            "fake_frames": int(result["fake_frames"]),
            "is_spoofed": bool(result["fake_percentage"] > 50),
            "frame_predictions": result["prediction"]
        }

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(SUPPORTED_FORMATS):
        raise HTTPException(status_code=400, detail="Only audio files are supported (.wav, .mp3, .flac, .ogg)")

    try:
        result = await process_single_file(file)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Accept multiple audio files and return inference results for each.
    Send with:  -F "files=@a.wav" -F "files=@b.wav" ...
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    results = []

    for file in files:
        if not file.filename.endswith(SUPPORTED_FORMATS):
            results.append({
                "filename": file.filename,
                "error": "Unsupported file format"
            })
            continue

        try:
            result = await process_single_file(file)
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {str(e)}")
            results.append({
                "filename": file.filename,
                "error": str(e)
            })

    return JSONResponse(content={
        "total_files": len(files),
        "processed": len([r for r in results if "error" not in r]),
        "failed": len([r for r in results if "error" in r]),
        "results": results
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
"""
FastAPI REST API for Deepfake Detection
========================================
Endpoints:
  POST /api/detect/image   - analyze uploaded image
  POST /api/detect/video   - analyze uploaded video
  GET  /api/health         - health check
  GET  /api/stats          - detection statistics
"""

import io
import time
import base64
import logging
from pathlib import Path
from typing import Optional
from collections import defaultdict

import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from src.inference import DeepfakeInference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App Setup ──────────────────────────────────────────────
app = FastAPI(
    title="Deepfake Detection API",
    description="AI-powered deepfake detection using EfficientNet + Frequency Analysis",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Model ─────────────────────────────────────────────
MODEL_PATH = Path("checkpoints/best_model.pth")
engine = DeepfakeInference(
    model_path=str(MODEL_PATH) if MODEL_PATH.exists() else None,
    device="cuda" if __import__("torch").cuda.is_available() else "cpu",
    generate_heatmap=True,
)
logger.info("Model loaded successfully.")

# ── Stats ──────────────────────────────────────────────────
stats = defaultdict(int)
stats["total"] = 0
stats["fake"]  = 0
stats["real"]  = 0
stats["errors"]= 0


# ── Pydantic Models ────────────────────────────────────────

class ImageDetectionResponse(BaseModel):
    label: str
    probability: float
    confidence: float
    is_fake: bool
    heatmap_b64: Optional[str] = None
    processing_time_ms: float


class VideoDetectionResponse(BaseModel):
    label: str
    probability: float
    confidence: float
    is_fake: bool
    frames_analyzed: int
    frame_results: list
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool


class StatsResponse(BaseModel):
    total_analyzed: int
    fake_detected: int
    real_detected: int
    errors: int
    fake_rate: float


# ── Helpers ────────────────────────────────────────────────

def pil_to_b64(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def ndarray_to_b64(arr: np.ndarray) -> str:
    img = Image.fromarray(arr.astype(np.uint8))
    return pil_to_b64(img)


# ── Endpoints ──────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
async def health():
    return {
        "status": "ok",
        "device": str(engine.device),
        "model_loaded": True,
    }


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    total = stats["total"]
    fake  = stats["fake"]
    return {
        "total_analyzed": total,
        "fake_detected": fake,
        "real_detected": stats["real"],
        "errors": stats["errors"],
        "fake_rate": fake / total if total > 0 else 0.0,
    }


@app.post("/api/detect/image", response_model=ImageDetectionResponse)
async def detect_image(
    file: UploadFile = File(...),
    crop_face: bool = True,
    return_heatmap: bool = True,
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    t0 = time.time()
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        result = engine.predict_image(img, crop_face=crop_face)

        heatmap_b64 = None
        if return_heatmap and result.heatmap is not None:
            heatmap_b64 = ndarray_to_b64(result.heatmap)

        stats["total"] += 1
        stats["fake" if result.is_fake else "real"] += 1

        return {
            **result.to_dict(),
            "heatmap_b64": heatmap_b64,
            "processing_time_ms": (time.time() - t0) * 1000,
        }
    except Exception as e:
        stats["errors"] += 1
        logger.error(f"Image detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detect/video", response_model=VideoDetectionResponse)
async def detect_video(
    file: UploadFile = File(...),
    frame_interval: int = 10,
    max_frames: int = 30,
    crop_face: bool = True,
):
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video.")

    t0 = time.time()
    tmp_path = Path(f"/tmp/upload_{int(time.time())}_{file.filename}")
    try:
        contents = await file.read()
        tmp_path.write_bytes(contents)

        result = engine.predict_video(
            str(tmp_path),
            frame_interval=frame_interval,
            max_frames=max_frames,
            crop_face=crop_face,
        )

        stats["total"] += 1
        stats["fake" if result.is_fake else "real"] += 1

        return {
            **result.to_dict(),
            "frames_analyzed": len(result.frame_results),
            "frame_results": result.frame_results,
            "processing_time_ms": (time.time() - t0) * 1000,
        }
    except Exception as e:
        stats["errors"] += 1
        logger.error(f"Video detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# Mount static files for frontend
static_dir = Path("static")
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

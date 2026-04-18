"""
OCR Service FastAPI application.
Accepts image uploads, segments digits, runs CNN inference, returns recognized text.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import torch
from PIL import Image, UnidentifiedImageError

from model import DigitCNN, NUM_CLASSES
from segment import segment_digits

app = FastAPI(title="OCR Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model.pt')
METRICS_PATH = os.path.join(os.path.dirname(__file__), '..', 'eval_metrics.json')

# Device selection
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load model at startup
model = DigitCNN().to(device)
model_loaded = False

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    model_loaded = True


# Pydantic response model
class OCRResponse(BaseModel):
    text: str
    per_digit_confidence: list[float]
    digit_count: int
    latency_ms: float


@app.get("/healthz")
def health():
    return {"status": "ok", "model_loaded": model_loaded}


@app.post("/ocr", response_model=OCRResponse)
async def ocr(file: UploadFile = File(...)):
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run train.py first."
        )

    # Read and parse image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except (UnidentifiedImageError, Exception) as e:
        raise HTTPException(status_code=422, detail=f"Invalid image file: {str(e)}")

    start_time = time.time()

    # Segment digits
    digit_tensors = segment_digits(image)

    if not digit_tensors:
        latency_ms = (time.time() - start_time) * 1000
        return OCRResponse(
            text="",
            per_digit_confidence=[],
            digit_count=0,
            latency_ms=round(latency_ms, 2),
        )

    # Batch all tensors into a single forward pass
    # Each tensor is (1, 28, 28); stack to (N, 1, 28, 28)
    batch = torch.stack(digit_tensors).to(device)

    with torch.no_grad():
        logits = model(batch)  # (N, 10)
        probs = torch.softmax(logits, dim=1)  # (N, 10)
        predicted_classes = probs.argmax(dim=1)  # (N,)
        confidences = probs.max(dim=1).values  # (N,)

    latency_ms = (time.time() - start_time) * 1000

    predicted_digits = predicted_classes.cpu().tolist()
    confidence_list = [round(float(c), 4) for c in confidences.cpu().tolist()]

    text = " ".join(str(d) for d in predicted_digits)

    return OCRResponse(
        text=text,
        per_digit_confidence=confidence_list,
        digit_count=len(predicted_digits),
        latency_ms=round(latency_ms, 2),
    )


@app.get("/metrics")
def metrics():
    if not os.path.exists(METRICS_PATH):
        return {"error": "not trained yet"}
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


@app.get("/test-sample")
def test_sample(n: int = 8):
    """Generate a random MNIST digit strip with known ground-truth labels."""
    import random, base64
    import numpy as np
    from torchvision import datasets, transforms
    from PIL import Image as PILImage

    data_root = os.path.join(os.path.dirname(__file__), '..', 'data')
    val_ds = datasets.MNIST(
        root=data_root, train=False, download=False,
        transform=transforms.Compose([transforms.ToTensor()])
    )
    indices = random.sample(range(len(val_ds)), min(n, len(val_ds)))
    digit_size = 28  # native MNIST size — no resize, no quality loss
    gap = 8
    strip_w = digit_size * len(indices) + gap * (len(indices) - 1)
    strip = PILImage.new('L', (strip_w, digit_size), color=0)  # black background (MNIST format)

    labels = []
    for i, idx in enumerate(indices):
        img_t, label = val_ds[idx]
        arr = (img_t.squeeze().numpy() * 255).astype(np.uint8)  # white digit on black — native MNIST
        digit_img = PILImage.fromarray(arr)  # no resize — keep 28x28 intact
        strip.paste(digit_img, (i * (digit_size + gap), 0))
        labels.append(int(label))

    buf = io.BytesIO()
    strip.save(buf, format='PNG')
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    return {"image_b64": img_b64, "ground_truth": " ".join(str(l) for l in labels)}


@app.get("/live-accuracy")
def live_accuracy():
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from augment import apply_gaussian, apply_salt_and_pepper

    data_root = os.path.join(os.path.dirname(__file__), '..', 'data')
    transform = transforms.Compose([transforms.ToTensor()])
    val_ds = datasets.MNIST(root=data_root, train=False, download=False, transform=transform)
    loader = DataLoader(val_ds, batch_size=500, shuffle=True)
    x, y = next(iter(loader))

    model.eval()
    with torch.no_grad():
        y_dev = y.to(device)
        acc_clean = (model(x.to(device)).argmax(dim=1) == y_dev).float().mean().item()
        acc_gaussian = (model(apply_gaussian(x).to(device)).argmax(dim=1) == y_dev).float().mean().item()
        acc_sp = (model(apply_salt_and_pepper(x).to(device)).argmax(dim=1) == y_dev).float().mean().item()

    return {
        "accuracy_clean": round(acc_clean, 4),
        "accuracy_gaussian": round(acc_gaussian, 4),
        "accuracy_salt_and_pepper": round(acc_sp, 4),
    }

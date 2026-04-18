import base64
import math
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from fgk import encode, decode

app = FastAPI(title="Huffman Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class CompressRequest(BaseModel):
    text: str


class CompressResponse(BaseModel):
    compressed_b64: str
    original_bytes: int
    compressed_bytes: int
    ratio: float
    entropy: float
    efficiency: float
    latency_ms: float


class DecompressRequest(BaseModel):
    compressed_b64: str


class DecompressResponse(BaseModel):
    text: str
    latency_ms: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_entropy(text: str) -> float:
    """Shannon entropy in bits per symbol over the input text characters."""
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/healthz")
def health() -> dict:
    return {"status": "ok"}


@app.post("/compress", response_model=CompressResponse)
def compress(req: CompressRequest) -> CompressResponse:
    text = req.text

    t0 = time.perf_counter()
    try:
        compressed = encode(text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    latency_ms = (time.perf_counter() - t0) * 1000

    compressed_b64 = base64.b64encode(compressed).decode("ascii")
    original_bytes = len(text.encode("latin-1")) if text else 0
    compressed_bytes = len(compressed)

    if not text:
        return CompressResponse(
            compressed_b64=compressed_b64,
            original_bytes=0,
            compressed_bytes=compressed_bytes,
            ratio=0.0,
            entropy=0.0,
            efficiency=0.0,
            latency_ms=latency_ms,
        )

    ratio = compressed_bytes / original_bytes
    entropy = _compute_entropy(text)
    avg_bits_per_symbol = (compressed_bytes * 8) / original_bytes

    # Single unique char → entropy=0, but FGK still achieves minimum; efficiency=1.0 by convention
    if entropy == 0.0:
        efficiency = 1.0
    else:
        efficiency = entropy / avg_bits_per_symbol

    return CompressResponse(
        compressed_b64=compressed_b64,
        original_bytes=original_bytes,
        compressed_bytes=compressed_bytes,
        ratio=ratio,
        entropy=entropy,
        efficiency=efficiency,
        latency_ms=latency_ms,
    )


@app.post("/decompress", response_model=DecompressResponse)
def decompress(req: DecompressRequest) -> DecompressResponse:
    try:
        compressed = base64.b64decode(req.compressed_b64, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid base64: {exc}")

    t0 = time.perf_counter()
    try:
        text = decode(compressed)
    except (ValueError, EOFError) as exc:
        raise HTTPException(status_code=422, detail=f"Corrupt Huffman stream: {exc}")
    latency_ms = (time.perf_counter() - t0) * 1000

    return DecompressResponse(text=text, latency_ms=latency_ms)

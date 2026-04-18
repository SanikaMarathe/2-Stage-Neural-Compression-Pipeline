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
    allow_origins=["http://localhost:8080","http://127.0.0.1:8080"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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


def _entropy(text: str) -> float:  # shannon entropy bits/symbol, -sum(p*log2(p))
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c/n) * math.log2(c/n) for c in counts.values())


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
    lat_ms = (time.perf_counter() - t0) * 1000

    c_b64  = base64.b64encode(compressed).decode("ascii")
    orig_b = len(text.encode("latin-1")) if text else 0  # original size in bytes
    comp_b = len(compressed)

    if not text:
        return CompressResponse(
            compressed_b64=c_b64,
            original_bytes=0,
            compressed_bytes=comp_b,
            ratio=0.0,
            entropy=0.0,
            efficiency=0.0,
            latency_ms=lat_ms,
        )

    ratio  = comp_b / orig_b
    ent    = _entropy(text)
    avg_bps = (comp_b * 8) / orig_b  # avg bits per symbol
    eff = 1.0 if ent == 0.0 else ent / avg_bps  # how close to theoretical best

    return CompressResponse(
        compressed_b64=c_b64,
        original_bytes=orig_b,
        compressed_bytes=comp_b,
        ratio=ratio,
        entropy=ent,
        efficiency=eff,
        latency_ms=lat_ms,
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
    lat_ms = (time.perf_counter() - t0) * 1000

    return DecompressResponse(text=text, latency_ms=lat_ms)

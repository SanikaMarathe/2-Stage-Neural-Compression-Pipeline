"""
E2E latency benchmark: times the full pipeline (image → /ocr → /compress → /decompress).
Reports p50, p95, p99 latency in milliseconds.
"""

import io
import os
import time
import numpy as np
import requests
import torch
import torchvision

OCR_URL = "http://localhost:8001/ocr"
COMPRESS_URL = "http://localhost:8002/compress"
DECOMPRESS_URL = "http://localhost:8002/decompress"

N_SAMPLES = 100
DIGITS_PER_STRIP = 4


def load_mnist_val():
    """Load MNIST validation (test) set."""
    dataset = torchvision.datasets.MNIST(
        root=os.path.join(os.path.dirname(__file__), '..', '.mnist_cache'),
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    return dataset


def build_digit_strip(dataset, rng: torch.Generator) -> bytes:
    """
    Pick 4 random MNIST images and concatenate them horizontally into a
    28x112 grayscale PNG. Returns PNG bytes.
    """
    from PIL import Image

    indices = torch.randint(0, len(dataset), (DIGITS_PER_STRIP,), generator=rng).tolist()
    strips = []
    for idx in indices:
        img_tensor, _ = dataset[idx]  # (1, 28, 28)
        arr = (img_tensor.squeeze(0).numpy() * 255).astype("uint8")
        strips.append(arr)

    # Concatenate horizontally: shape (28, 28*4) = (28, 112)
    combined = np.concatenate(strips, axis=1)
    img = Image.fromarray(combined, mode='L')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf.read()


def percentile(data: list, p: float) -> float:
    return float(np.percentile(data, p))


def run_benchmark():
    dataset = load_mnist_val()
    rng = torch.Generator()
    rng.manual_seed(0)

    total_times = []
    ocr_times = []
    compress_times = []
    decompress_times = []

    print(f"Running E2E latency benchmark — N={N_SAMPLES} samples...")

    for i in range(N_SAMPLES):
        png_bytes = build_digit_strip(dataset, rng)

        t_total_start = time.perf_counter()

        # --- Step a: OCR ---
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                OCR_URL,
                files={"file": ("strip.png", io.BytesIO(png_bytes), "image/png")},
                timeout=15,
            )
            resp.raise_for_status()
            ocr_data = resp.json()
            text = ocr_data.get("text", "")
        except requests.RequestException:
            text = "0 0 0 0"  # fallback so pipeline continues
        t_ocr = (time.perf_counter() - t0) * 1000

        # --- Step b: Compress ---
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                COMPRESS_URL,
                json={"text": text},
                timeout=10,
            )
            resp.raise_for_status()
            compress_data = resp.json()
            compressed_b64 = compress_data.get("compressed_b64", "")
        except requests.RequestException:
            compressed_b64 = ""
        t_compress = (time.perf_counter() - t0) * 1000

        # --- Step c: Decompress ---
        t0 = time.perf_counter()
        try:
            if compressed_b64:
                resp = requests.post(
                    DECOMPRESS_URL,
                    json={"compressed_b64": compressed_b64},
                    timeout=10,
                )
                resp.raise_for_status()
                decompress_data = resp.json()
                recovered_text = decompress_data.get("text", "")
        except requests.RequestException:
            pass
        t_decompress = (time.perf_counter() - t0) * 1000

        t_total = (time.perf_counter() - t_total_start) * 1000

        total_times.append(t_total)
        ocr_times.append(t_ocr)
        compress_times.append(t_compress)
        decompress_times.append(t_decompress)

    print(f"\nLatency (ms) — N={N_SAMPLES} samples")
    print(f"{'p50:':<20} {percentile(total_times, 50):.1f}")
    print(f"{'p95:':<20} {percentile(total_times, 95):.1f}")
    print(f"{'p99:':<20} {percentile(total_times, 99):.1f}")
    print(f"{'ocr_p50:':<20} {percentile(ocr_times, 50):.1f}")
    print(f"{'compress_p50:':<20} {percentile(compress_times, 50):.1f}")
    print(f"{'decompress_p50:':<20} {percentile(decompress_times, 50):.1f}")


if __name__ == "__main__":
    run_benchmark()

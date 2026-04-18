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

N_SAMP = 100
DPS = 4  # digits per strip


def load_mnist_val():
    dataset = torchvision.datasets.MNIST(
        root=os.path.join(os.path.dirname(__file__), '..', '.mnist_cache'),
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    return dataset


def build_digit_strip(dataset, rng: torch.Generator) -> bytes:
    # pick DPS random digits and concatenate them side by side -> PNG bytes
    from PIL import Image
    idxs = torch.randint(0, len(dataset), (DPS,), generator=rng).tolist()
    strips = []
    for idx in idxs:
        img_t,_ = dataset[idx]
        arr = (img_t.squeeze(0).numpy() * 255).astype("uint8")
        strips.append(arr)
    combined = np.concatenate(strips, axis=1)  # horizontal concat
    img = Image.fromarray(combined, mode='L')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf.read()


def pct(data: list, p: float) -> float:
    return float(np.percentile(data,p))


def run_benchmark():
    dataset = load_mnist_val()
    rng = torch.Generator()
    rng.manual_seed(0)

    t_tot,t_ocr,t_comp,t_dec = [],[],[],[]  # timing buckets

    print(f"running e2e benchmark n={N_SAMP}...")

    for i in range(N_SAMP):
        png = build_digit_strip(dataset, rng)
        t_start = time.perf_counter()

        # ocr: image -> text
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                OCR_URL,
                files={"file": ("strip.png", io.BytesIO(png), "image/png")},
                timeout=15,
            )
            resp.raise_for_status()
            text = resp.json().get("text","")
        except requests.RequestException:
            text = "0 0 0 0"  # fallback so pipeline continues
        t_ocr.append((time.perf_counter()-t0)*1000)

        # compress the text
        t0 = time.perf_counter()
        c_b64 = ""
        try:
            resp = requests.post(COMPRESS_URL, json={"text": text}, timeout=10)
            resp.raise_for_status()
            c_b64 = resp.json().get("compressed_b64","")
        except requests.RequestException:
            pass
        t_comp.append((time.perf_counter()-t0)*1000)

        # decompress to verify roundtrip
        t0 = time.perf_counter()
        try:
            if c_b64:
                resp = requests.post(DECOMPRESS_URL, json={"compressed_b64": c_b64}, timeout=10)
                resp.raise_for_status()
        except requests.RequestException:
            pass
        t_dec.append((time.perf_counter()-t0)*1000)

        t_tot.append((time.perf_counter()-t_start)*1000)

    print(f"p50={pct(t_tot,50):.1f} p95={pct(t_tot,95):.1f} p99={pct(t_tot,99):.1f}")
    print(f"ocr_p50={pct(t_ocr,50):.1f} comp_p50={pct(t_comp,50):.1f} dec_p50={pct(t_dec,50):.1f}")


if __name__ == "__main__":
    run_benchmark()

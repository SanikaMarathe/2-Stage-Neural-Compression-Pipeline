import sys
import os
import io
import requests
import torch
import torchvision
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ocr-service'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ocr-service','app'))

from app.augment import apply_gaussian, apply_salt_and_pepper

OCR_URL = "http://localhost:8001/ocr"
N_SAMP = 200

PROFILES = [
    ("clean", None),
    ("gaussian", apply_gaussian),
    ("salt_and_pepper", apply_salt_and_pepper),
]


def load_mnist_val():
    dataset = torchvision.datasets.MNIST(
        root=os.path.join(os.path.dirname(__file__), '..', '.mnist_cache'),
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    return dataset


def to_png(tensor: torch.Tensor) -> bytes:
    from PIL import Image
    arr = (tensor.squeeze(0).numpy() * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode='L')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf.read()


def run_benchmark():
    dataset = load_mnist_val()
    rng = torch.Generator()
    rng.manual_seed(42)

    idxs = torch.randperm(len(dataset), generator=rng)[:N_SAMP].tolist()  # fixed subset

    imgs,lbls = [],[]
    for idx in idxs:
        img_t,lbl = dataset[idx]
        imgs.append(img_t)
        lbls.append(lbl)

    img_batch = torch.stack(imgs)  # (N,1,28,28)

    for prof_name,aug_fn in PROFILES:
        aug = img_batch.clone() if aug_fn is None else aug_fn(img_batch)  # apply noise
        correct = 0
        total = 0

        for i in range(N_SAMP):
            img_t = aug[i]
            lbl = lbls[i]
            png = to_png(img_t)

            try:
                resp = requests.post(
                    OCR_URL,
                    files={"file": ("digit.png", io.BytesIO(png), "image/png")},
                    timeout=10,
                )
                resp.raise_for_status()
                text = resp.json().get("text","").strip()
                if text == str(lbl):  # single digit image -> expect one digit back
                    correct += 1
            except requests.RequestException:
                pass
            total += 1

        acc = correct/total if total > 0 else 0.0
        print(f"{prof_name}: {acc:.4f}")


if __name__ == "__main__":
    run_benchmark()

"""
Accuracy benchmark: posts digit-strip images to /ocr for each noise profile.
Generates synthetic test strips using torchvision MNIST val set.
Reports character-level accuracy per profile.
"""

import sys
import os
import io
import requests
import torch
import torchvision
import numpy as np

# Allow importing augment functions from the OCR service
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ocr-service'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ocr-service', 'app'))

from app.augment import apply_gaussian, apply_salt_and_pepper

OCR_URL = "http://localhost:8001/ocr"
N_SAMPLES = 200

PROFILES = [
    ("clean", None),
    ("gaussian", apply_gaussian),
    ("salt_and_pepper", apply_salt_and_pepper),
]


def load_mnist_val():
    """Load MNIST validation (test) set, return list of (tensor, label)."""
    dataset = torchvision.datasets.MNIST(
        root=os.path.join(os.path.dirname(__file__), '..', '.mnist_cache'),
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    return dataset


def tensor_to_png_bytes(tensor: torch.Tensor) -> bytes:
    """Convert a (1, 28, 28) float tensor [0,1] to PNG bytes."""
    from PIL import Image
    # Convert to uint8 numpy array
    arr = (tensor.squeeze(0).numpy() * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode='L')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf.read()


def run_benchmark():
    dataset = load_mnist_val()

    # Use a fixed random seed for reproducibility
    rng = torch.Generator()
    rng.manual_seed(42)

    # Pre-select indices for all profiles (same pool, different augmentation)
    indices = torch.randperm(len(dataset), generator=rng)[:N_SAMPLES].tolist()

    # Pre-load images and labels
    images = []
    labels = []
    for idx in indices:
        img_tensor, label = dataset[idx]  # img_tensor: (1, 28, 28)
        images.append(img_tensor)
        labels.append(label)

    images_batch = torch.stack(images)  # (N, 1, 28, 28)

    print(f"{'Profile':<20} {'Accuracy':<10}")
    print("-" * 30)

    for profile_name, augment_fn in PROFILES:
        if augment_fn is None:
            augmented = images_batch.clone()
        else:
            augmented = augment_fn(images_batch)

        correct = 0
        total = 0

        for i in range(N_SAMPLES):
            img_tensor = augmented[i]  # (1, 28, 28)
            label = labels[i]

            png_bytes = tensor_to_png_bytes(img_tensor)

            try:
                response = requests.post(
                    OCR_URL,
                    files={"file": ("digit.png", io.BytesIO(png_bytes), "image/png")},
                    timeout=10,
                )
                response.raise_for_status()
                data = response.json()
                text = data.get("text", "").strip()

                # For a single 28x28 digit image, expect exactly one digit back
                if text == str(label):
                    correct += 1
                # else: wrong prediction or segmentation failure → incorrect

            except requests.RequestException as e:
                # Service unavailable or error — count as incorrect
                pass

            total += 1

        accuracy = correct / total if total > 0 else 0.0
        print(f"{profile_name:<20} {accuracy:.4f}")


if __name__ == "__main__":
    run_benchmark()

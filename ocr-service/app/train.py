"""
Training script for DigitCNN on MNIST.

Run from the ocr-service/ directory:
    python app/train.py

Outputs:
    ../model.pt          — saved model state dict (relative to this file → ocr-service/model.pt)
    ../eval_metrics.json — final evaluation metrics as JSON
"""

import json
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(__file__))
from model import DigitCNN
from augment import augment_batch, apply_gaussian, apply_salt_and_pepper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def evaluate(model: nn.Module, batches: list, device: torch.device) -> float:
    """Evaluate model accuracy on a pre-loaded list of (x, y) batches."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in batches:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Device selection
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Download + load MNIST
    # Training transform includes spatial augmentation to match segmentation pipeline distortions
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
    ])
    val_transform = transforms.Compose([transforms.ToTensor()])

    data_root = os.path.join(os.path.dirname(__file__), '..', 'data')
    train_ds = datasets.MNIST(root=data_root, train=True,  download=False, transform=train_transform)
    val_ds   = datasets.MNIST(root=data_root, train=False, download=False, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

    # 3. Three held-out val splits (created once, kept fixed)
    val_clean    = [batch for batch in DataLoader(val_ds, batch_size=256)]
    val_gaussian = [(apply_gaussian(x), y) for x, y in val_clean]
    val_sp       = [(apply_salt_and_pepper(x), y) for x, y in val_clean]

    # 4. Model, loss, optimizer
    model     = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    # 5. Training loop — 15 epochs with spatial + noise augmentation
    acc_clean = acc_gaussian = acc_sp = 0.0

    for epoch in range(1, 16):
        model.train()
        for images, labels in train_loader:
            images = augment_batch(images)          # random noise augmentation
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Evaluate on all 3 splits
        acc_clean    = evaluate(model, val_clean,    device)
        acc_gaussian = evaluate(model, val_gaussian, device)
        acc_sp       = evaluate(model, val_sp,       device)
        print(
            f"Epoch {epoch}: "
            f"clean={acc_clean:.4f} "
            f"gaussian={acc_gaussian:.4f} "
            f"sp={acc_sp:.4f}"
        )

    # 6. Save model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved → {os.path.abspath(model_path)}")

    # 7. Save metrics
    metrics = {
        "accuracy_clean":           acc_clean,
        "accuracy_gaussian":        acc_gaussian,
        "accuracy_salt_and_pepper": acc_sp,
        "epochs_trained":           15,
        "dataset":                  "MNIST",
        "num_classes":              10,
    }
    metrics_path = os.path.join(os.path.dirname(__file__), '..', 'eval_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved → {os.path.abspath(metrics_path)}")

    # 8. Final summary
    threshold = 0.95
    all_pass = all(a >= threshold for a in (acc_clean, acc_gaussian, acc_sp))
    status = "PASS" if all_pass else "FAIL"
    print(
        f"\nFinal: clean={acc_clean:.4f} gaussian={acc_gaussian:.4f} sp={acc_sp:.4f} "
        f"| ≥{threshold} threshold: {status}"
    )


if __name__ == '__main__':
    main()

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


def evaluate(model: nn.Module, batches: list, device: torch.device) -> float:
    # count correct predictions across all batches
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x,y in batches:
            x,y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    # train transform includes spatial augments to match segmentation distortions
    tr_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1,0.1), scale=(0.9,1.1), shear=5),
    ])
    v_tfm = transforms.Compose([transforms.ToTensor()])

    data_root = os.path.join(os.path.dirname(__file__), '..', 'data')
    tr_ds = datasets.MNIST(root=data_root, train=True,  download=False, transform=tr_tfm)
    v_ds  = datasets.MNIST(root=data_root, train=False, download=False, transform=v_tfm)

    tr_ldr = DataLoader(tr_ds, batch_size=128, shuffle=True)
    v_cln  = [batch for batch in DataLoader(v_ds, batch_size=256)]  # preload val
    v_gaus = [(apply_gaussian(x),y) for x,y in v_cln]               # noisy copies
    v_sp   = [(apply_salt_and_pepper(x),y) for x,y in v_cln]

    model     = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    ac = ag = asp = 0.0

    for epoch in range(1,16):
        model.train()
        for images,labels in tr_ldr:
            images = augment_batch(images)  # random noise per image
            images,labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        sched.step()  # decay lr

        ac  = evaluate(model, v_cln, device)
        ag  = evaluate(model, v_gaus, device)
        asp = evaluate(model, v_sp, device)
        print(f"ep{epoch} c={ac:.3f} g={ag:.3f} sp={asp:.3f}")

    mpath = os.path.join(os.path.dirname(__file__), '..', 'model.pt')
    torch.save(model.state_dict(), mpath)
    print(f"saved {os.path.abspath(mpath)}")

    metrics = {
        "accuracy_clean": ac,
        "accuracy_gaussian": ag,
        "accuracy_salt_and_pepper": asp,
        "epochs_trained": 15,
        "dataset": "MNIST",
        "num_classes": 10,
    }
    mpath2 = os.path.join(os.path.dirname(__file__), '..', 'eval_metrics.json')
    with open(mpath2,'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"metrics -> {os.path.abspath(mpath2)}")

    thr = 0.95
    ok = all(a >= thr for a in (ac,ag,asp))  # fail if any split below 95%
    print(f"final: c={ac:.4f} g={ag:.4f} sp={asp:.4f} | {'PASS' if ok else 'FAIL'}")


if __name__ == '__main__':
    main()

"""
DigitCNN: A CNN for handwritten digit recognition on 28x28 grayscale images.
Trained on MNIST (10 classes: digits 0–9).
"""

import torch
import torch.nn as nn

NUM_CLASSES = 10  # MNIST digit classes


class DigitCNN(nn.Module):
    """
    Two-block CNN for MNIST digit classification.

    Input:  (N, 1, 28, 28)
    Output: (N, 10) — raw logits, no softmax
    """

    def __init__(self):
        super().__init__()

        self.conv_block = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                  # → 32×14×14
            nn.Dropout(p=0.25),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                  # → 64×7×7
            nn.Dropout(p=0.25),
        )

        self.fc_block = nn.Sequential(
            nn.Flatten(),                  # → 3136
            nn.Linear(3136, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, NUM_CLASSES),   # → 10 logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block(x)
        x = self.fc_block(x)
        return x

    def predict(self, x: torch.Tensor) -> list:
        """Return predicted class indices as a list of ints (argmax over logits)."""
        with torch.no_grad():
            logits = self(x)
        return logits.argmax(dim=1).tolist()


def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

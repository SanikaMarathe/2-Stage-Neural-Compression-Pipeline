import torch
import torch.nn as nn

NUM_CLASSES=10  # mnist digit classes


class DigitCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # two conv blocks, each halves spatial size via maxpool
        self.conv_block = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 32x14x14
            nn.Dropout(p=0.25),

            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 64x7x7
            nn.Dropout(p=0.25),
        )

        # flatten then classify
        self.fc_block = nn.Sequential(
            nn.Flatten(),  # 3136
            nn.Linear(3136,128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128,NUM_CLASSES),  # raw logits out
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block(x)
        x = self.fc_block(x)
        return x

    def predict(self, x: torch.Tensor) -> list:
        # returns class indices, no grad needed
        with torch.no_grad():
            logits = self(x)
        return logits.argmax(dim=1).tolist()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

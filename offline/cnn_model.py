import torch
import torch.nn as nn


class TremorCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=7, padding=3),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(-1)
        return self.classifier(x)


if __name__ == "__main__":
    model = TremorCNN()
    x = torch.randn(1, 3, 256)
    y = model(x)

    print("Model:", model.__class__.__name__)
    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(y.shape))
    print("Forward pass:", "PASS" if y.shape == (1, 2) else "FAIL")

import torch.nn as nn


class SimpleCnn(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 8, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 96, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.out = nn.Linear(96 * 5 * 5, n_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = nn.Flatten()(x)
        logits = self.out(x)
        return logits
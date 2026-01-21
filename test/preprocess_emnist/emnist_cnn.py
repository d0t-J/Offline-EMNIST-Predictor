import torch
import torch.nn as nn


class EMNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 62)

    def forward(self, x):
        # print(f"Input: {x.shape}")

        x = self.conv1(x)
        # print(f"Shape after conv1: {x.shape}")

        x = self.conv2(x)
        # print(f"Shape after conv2: {x.shape}")

        x = x.view(x.size(0), -1)
        # print(f"After flattening: {x.shape}")

        x = torch.relu(self.fc1(x))
        # print(f"After FC1: {x.shape}")

        x = self.fc2(x)
        # print(f"After FC2: {x.shape}")

        return x


if __name__ == "__main__":
    model = EMNIST_CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    dummy = torch.randn(1, 1, 28, 28)
    out = model(dummy)

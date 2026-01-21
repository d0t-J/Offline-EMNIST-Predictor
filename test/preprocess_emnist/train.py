import torch
import torch.nn as nn

from emnist_cnn import EMNIST_CNN
from preprocess_emnist import EMNIST_Preprocessor

from torchvision import datasets
from torch.utils.data import DataLoader, random_split


def full_train():
    full_train_dataset = datasets.EMNIST(
        root="../EMNIST_exploration/data",
        split="byclass",
        train=True,
        download=False,
        transform=EMNIST_Preprocessor(),
    )

    train_size = int(0.9 * len(full_train_dataset))
    val_size = int(0.1 * len(full_train_dataset))

    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True
    )

    total_classes = len(full_train_dataset.classes)

    model = EMNIST_CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for batch_idx, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        print(f"Batch: {batch_idx}, Loss: {loss.item()}")
        # Remove this for real training
        break

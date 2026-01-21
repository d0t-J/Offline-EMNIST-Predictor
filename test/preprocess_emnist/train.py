import torch
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore")


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
    val_size = len(full_train_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=False
    )

    total_classes = len(full_train_dataset.classes)

    model = EMNIST_CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch: {epoch}/{num_epochs}")
        model.train()
        running_loss = 0.0

        for batch_idx, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 500 == 0:
                print(f"Batch: {batch_idx + 1}, Loss: {loss.item()}")
                running_loss = 0.0

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x)
            loss = criterion(logits, y)
            val_loss += loss.item() * x.size(0)

            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    val_loss /= total
    val_acc = correct / total
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

    torch.save(model.state_dict(), "emnist_cnn.pth")


def main():
    full_train()


if __name__ == "__main__":
    main()

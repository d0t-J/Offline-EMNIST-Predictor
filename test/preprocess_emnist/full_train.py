from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from preprocess_emnist import EMNISTPreprocessor


def full_train():
    full_train_dataset = datasets.EMNIST(
        root="../EMNIST_exploration/data",
        split="byclass",
        train=True,
        download=False,
        transform=EMNISTPreprocessor(),
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

    total_classes = full_train_dataset.classes



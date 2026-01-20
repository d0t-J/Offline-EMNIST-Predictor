import torch
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

transform = transforms.ToTensor()

dataset = datasets.EMNIST(
    root="./data", split="byclass", train=True, download=True, transform=transform
)

print(f"Total Samples {len(dataset)}")

image, label = dataset[0]

print(f"Image Shape: {image.shape}")
print(f"Label: {label}")
print(f"Pixel Min: {image.min().item()}\tPixel Max: {image.max().item()}")

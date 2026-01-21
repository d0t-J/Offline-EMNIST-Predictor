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

fig, axes = plt.subplots(2, 5, figsize=(10, 4))

for idx, ax in enumerate(axes.flat):
    img, lbl = dataset[idx]
    ax.imshow(img.squeeze(), cmap="gray")
    ax.set_title(f"Label: {lbl}")
    ax.axis("off")

plt.show()

from torchvision import datasets, transforms
import json

dataset = datasets.EMNIST(
    root="../EMNIST_exploration/data",
    split="byclass",
    train=True,
    download=False,
    transform=transforms.ToTensor(),
)

with open("class_vocab.json", "w") as f:
    json.dump(dataset.classes, f)

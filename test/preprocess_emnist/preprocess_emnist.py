import matplotlib.pyplot as plt

from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
from PIL import Image, ImageOps


# ! In order to correct the orientation, we Flip the images Horizontally
# ! Then we rotate the images 90 degrees anti-clockwise
class EMNISTPreprocessor:
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.Lambda(self.fix_orientation_pil),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    @staticmethod
    def fix_orientation_pil(pil_img: Image.Image) -> Image.Image:
        # 1) Mirror horizontally (equivalent to RandomHorizontalFlip(p=1.0))
        flipped = ImageOps.mirror(pil_img)
        # 2) Rotate 90 degrees anti-clockwise (PIL rotate is counter-clockwise)
        rotated = flipped.rotate(90, expand=True)
        # If rotate changed size, optionally resize back to 28x28:
        if rotated.size != (28, 28):
            rotated = rotated.resize((28, 28))
        return rotated

    def __call__(self, pil_img: Image.Image):
        return self.transform(pil_img)


print("imports successful and class created")


# ! For demonstration purposes
def show_side_by_side(n=10):
    # raw transform (no orientation fix) for comparison
    raw_transform = transforms.ToTensor()
    fixed = EMNISTPreprocessor()

    raw_dataset = datasets.EMNIST(
        root="../EMNIST_exploration/data",
        split="byclass",
        train=True,
        download=False,
        transform=raw_transform,
    )
    fixed_dataset = datasets.EMNIST(
        root="../EMNIST_exploration/data",
        split="byclass",
        train=True,
        download=False,
        transform=fixed,
    )

    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4))
    for i in range(n):
        raw_img, raw_lbl = raw_dataset[i]
        fix_img, fix_lbl = fixed_dataset[i]

        axes[0, i].imshow(raw_img.squeeze(), cmap="gray")
        axes[0, i].set_title(f"raw idx:{i}\nlbl:{raw_lbl}")
        axes[0, i].axis("off")

        axes[1, i].imshow(fix_img.squeeze(), cmap="gray")
        axes[1, i].set_title(f"fixed idx:{i}\nlbl:{fix_lbl}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_side_by_side(n=10)

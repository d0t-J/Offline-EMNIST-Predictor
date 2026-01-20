import torch
from torchvision import transforms


class EMNISTPreprocessor:
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(self.fix_orientation),
            ]
        )

    @staticmethod
    def fix_orientation(img: torch.tensor) -> torch.tensor:
        img = torch.rot90(img, k=1, dims=[1, 2])
        img = torch.flip(img, dims=[2])

        return img

    def __call__(self, img):
        return self.transform(img)

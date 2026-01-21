import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from PIL import Image, ImageOps


class EMNIST_Preprocessor:
    def __init__(self):
        # ! In order to correct the orientation, we Flip the images Horizontally
        # ! Then we rotate the images 90 degrees anti-clockwise
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
        # 2) Rotate 90 degrees anti-clockwise (PIL img rotate is counter-clockwise)
        rotated = flipped.rotate(90, expand=True)
        # If rotate changed size, optionally resize back to 28x28:
        if rotated.size != (28, 28):
            rotated = rotated.resize((28, 28))
        return rotated

    def __call__(self, pil_img: Image.Image):
        return self.transform(pil_img)

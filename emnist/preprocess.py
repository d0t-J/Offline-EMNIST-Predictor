from torchvision import datasets, transforms
from PIL import Image, ImageOps


class EMNIST_OrientationFix:
    @staticmethod
    def fix_orientation_pil(pil_img: Image.Image) -> Image.Image:
        flipped = ImageOps.mirror(pil_img)
        rotated = flipped.rotate(90, expand=True)

        if rotated.size != (28, 28):
            rotated = rotated.resize((28, 28))
        return rotated


class EMNIST_Preprocessor:
    def __init__(self):
        # ! In order to correct the orientation, we Flip the images Horizontally
        # ! Then we rotate the images 90 degrees anti-clockwise
        self.transform = transforms.Compose(
            [
                transforms.Lambda(self.EMNIST_OrientationFix.fix_orientation_pil),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def __call__(self, pil_img: Image.Image):
        return self.transform(pil_img)


class EMNIST_Normalizer:
    def __init__(self):
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def __call__(self, pil_img: Image.Image):
        return self.transform(pil_img)

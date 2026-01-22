import sys
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageOps
from scipy import ndimage

from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
from PyQt5.QtGui import QPainter, QPen, QPixmap, QImage
from PyQt5.QtCore import Qt, QPoint

from emnist_inference import EMNIST_Inference


class Canvas(QWidget):
    def __init__(self):
        super().__init__()

        self.setFixedSize(280, 280)

        self.pixmap = QPixmap(self.size())
        self.pixmap.fill(Qt.black)

        self.last_point = QPoint()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            painter = QPainter(self.pixmap)
            pen = QPen(Qt.white, 15, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()

    def clear(self):
        self.pixmap.fill(Qt.black)
        self.update()

    def capture_image(self) -> Image.Image:
        """
        Capture canvas as a PIL Image.
        """
        qimage = self.pixmap.toImage().convertToFormat(QImage.Format_RGB888)

        width = qimage.width()
        height = qimage.height()

        ptr = qimage.bits()
        ptr.setsize(height * width * 3)

        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
        return Image.fromarray(arr)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Handwriting Canvas")

        self.canvas = Canvas()
        self.infer = EMNIST_Inference(
            model_path="emnist_cnn.pth", vocab_path="class_vocab.json", device="cpu"
        )

        self.btn_inspect = QPushButton("Inspect")
        self.btn_clear = QPushButton("Clear")

        self.btn_inspect.clicked.connect(self.inspect_canvas)
        self.btn_clear.clicked.connect(self.canvas.clear)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.btn_inspect)
        layout.addWidget(self.btn_clear)

        self.setLayout(layout)

    def inspect_canvas(self):
        raw = self.canvas.capture_image()

        pre = CanvasPreprocessor()
        processed = pre.preprocess(raw)

        predicted_char, confidence, _ = self.infer.predict(processed)

        processed.save("./test_canvas.png")
        print("Saved test_canvas.png")

        plt.figure(figsize=(6, 3))
        plt.subplot(1, 3, 1)
        plt.imshow(raw)
        plt.title("Raw Canvas")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(processed, cmap="gray")
        plt.title("Processed 28x28")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.text(
            0.5,
            0.5,
            f"Pred: {predicted_char}\nConf: {confidence:.2f}",
            fontsize=20,
            ha="center",
        )
        plt.axis("off")

        plt.tight_layout()
        plt.show()


class CanvasPreprocessor:
    def __init__(self, threshold=20):
        self.threshold = threshold

    def preprocess(self, img: Image.Image) -> Image.Image:
        """
        Convert raw canvas image to EMNIST-style 28x28 grayscale image.
        EMNIST training data: white strokes on black background
        Canvas: white strokes on black background (same!) → NO inversion needed
        """
        # 1. Grayscale
        img = img.convert("L")

        # 2. NO inversion - canvas already matches EMNIST format (white on black)
        # img = ImageOps.invert(img)  # REMOVED

        # 3. Convert to numpy
        img_np = np.array(img)

        # 4. Threshold
        img_np[img_np < self.threshold] = 0

        # 5. Bounding box
        coords = np.column_stack(np.where(img_np > 0))
        if coords.size == 0:
            return Image.fromarray(np.zeros((28, 28), dtype=np.uint8))

        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1

        img_np = img_np[y0:y1, x0:x1]

        img_pil = Image.fromarray(img_np)
        h, w = img_np.shape

        if h > w:
            new_h = 20
            new_w = int(round(w * (20 / h)))
        else:
            new_w = 20
            new_h = int(round(h * (20 / w)))

        img_pil = img_pil.resize((new_w, new_h), Image.BILINEAR)
        img_np = np.array(img_pil)

        # ! Apply padding here
        pad_h = 28 - new_h
        pad_w = 28 - new_w

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        img_np = np.pad(
            img_np, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant"
        )

        cy, cx = ndimage.center_of_mass(img_np)
        if np.isnan(cx) or np.isnan(cy):
            return Image.fromarray(img_np)

        shift_y = int(round(14 - cy))
        shift_x = int(round(14 - cx))

        img_np = ndimage.shift(img_np, shift=(shift_y, shift_x), mode="constant")

        return Image.fromarray(img_np.astype(np.uint8))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

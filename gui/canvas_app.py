import sys
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QVBoxLayout
)
from PyQt5.QtGui import QPainter, QPen, QPixmap, QImage
from PyQt5.QtCore import Qt, QPoint


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
        img = self.canvas.capture_image()

        plt.imshow(img)
        plt.title("Raw Canvas Capture")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

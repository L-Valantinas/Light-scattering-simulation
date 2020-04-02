from utils import log
from utils.display.fullscreen.Screen import Screen
import numpy as np
from PyQt5.QtGui import QIcon, QPixmap, QKeyEvent
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt

class QtFullScreenWindow(QWidget):
    def __init__(self, screen_idx, screen_shape):
        super().__init__()
        self.setWindowTitle("FullScreen{:d}".format(screen_idx))
        self.setGeometry(0, 0, screen_shape[1], screen_shape[0])
        # self.resize(pixmap.width(),pixmap.height())
        # self.showFullScreen()
        self.show()

        # Prepare the bitmap
        image_array = np.zeros((*screen_shape[:2], 3))
        self.__raw_bytes_header = b'P6\n%i %i\n255\n' % (image_array.shape[1], image_array.shape[0])
        self.__image = QPixmap()
        self.__image.loadFromData(self.__raw_bytes_header + image_array.tobytes())

        self.__label = QLabel(self)
        vbox = QVBoxLayout()
        vbox.addWidget(self.__label, alignment=Qt.AlignTop)
        self.setLayout(vbox)

        self.show()

    def keyPressEvent(self, event: QKeyEvent):
        # Just for safety
        if event.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def show_image(self, image_array):
        self.__image.loadFromData(self.__raw_bytes_header + image_array.tobytes())
        self.__label.setPixmap(self.__image)
        self.__label.show()
        self.show()
        time.sleep(0.001)


class QtScreen(Screen):
    def __init__(self, screen_idx):
        super().__init__(screen_idx)

        self.__app = QApplication([])
        self.__window = QtFullScreenWindow(screen_idx, self.shape)

    def show(self, image_array):
        super().show(image_array)
        # Convert to uint8 array
        image_array = np.array(image_array, dtype=np.uint8)
        # Make 3D array
        while image_array.ndim < 3:
            image_array = np.expand_dims(image_array, axis=image_array.ndim)
        # If a 3-vector, interpret as a color
        if len(image_array) == 3:
            image_array.reshape((1, 1, 3))
        # If 2D input, interpret as grayscale
        if image_array.shape[2] == 1:
            image_array = np.repeat(image_array[:, :, 0:1], repeats=3, axis=2)
        # If color only, expand to all pixels
        if image_array.shape[0] == 1 and image_array.shape[1] == 1:
            image_array = np.ones((self.screen.height, self.screen.width, 1), dtype=np.uint8) * image_array

        image_array = np.zeros((*self.shape, 3))
        image_array[:-1:2, :-1:2, 1] = 255
        self.__window.show_image(image_array)

    def close(self):
        self.__window.destroy()


if __name__ == '__main__':
    import time

    fs = QtScreen(0)
    img = np.zeros((*fs.shape, 3), dtype=np.uint8)
    img[::2,::2,:] = 255
    img[1::2,1::2,:] = 255
    rng = range(0, fs.shape[1], 200)
    start_time = time.time()
    for idx in rng:
        img[:, idx, 1] = 255
        fs.show(img)

    total_time = time.time() - start_time
    frames_per_second = len(rng) / total_time
    log.info("FullScreen display at {:0.1f} frames per second.".format(frames_per_second))

    fs.close()

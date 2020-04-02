from utils import log
from utils.display.fullscreen.Screen import Screen
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


class TestScreen(Screen):
    def __init__(self, screen_idx, ax: Axes=None, shape=(150, 200)):
        super().__init__(screen_idx)

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        self.__shape = shape[:2]
        self.__axes = ax

        image = np.zeros((*self.shape, 3))
        self.__image = ax.imshow(image)

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

        # Copy bytes to window on full screen window
        self.__image.set_data(image_array)
        time.sleep(0.001)  # allow time for figure to update

    def close(self):
        plt.close(self.__axes.get_figure())


if __name__ == '__main__':
    import time

    fs = TestScreen(0)
    img = np.zeros((*fs.shape, 3), dtype=np.uint8)
    img[::2,::2,:] = 255
    img[1::2,1::2,:] = 255
    rng = range(0, fs.shape[1], 20)
    start_time = time.time()
    for idx in rng:
        img[:, idx, 1] = 255
        fs.show(img)

    total_time = time.time() - start_time
    frames_per_second = len(rng) / total_time
    log.info("FullScreen display at {:0.1f} frames per second.".format(frames_per_second))

    fs.close()
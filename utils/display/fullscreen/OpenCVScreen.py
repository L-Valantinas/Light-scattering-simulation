from utils import log
from utils.display.fullscreen.Screen import Screen
import numpy as np
import tkinter as tk
import cv2
import os


class OpenCVScreen(Screen):
    def __init__(self, screen_idx):
        super().__init__(screen_idx)

        os.environ['SDL_VIDEO_CENTERED'] = '1'

        self.__window_name = "FullScreen{:d}".format(screen_idx)

        # Make full screen
        cv2.namedWindow(self.__window_name, cv2.WND_PROP_FULLSCREEN)
        # cv2.moveWindow(self.__window_name, self.screen.x - 1, self.screen.y - 1)
        # cv2.resizeWindow(self.__window_name, self.screen.width + 2, self.screen.height + 2)
        cv2.setWindowProperty(self.__window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cv2.waitKey(1)

    @property
    def shape(self):
        # OpenCV always seems to show a 1 pixel border
        return (super().shape[0] - 2, super().shape[1] - 2)

    def show(self, image_array):
        super().show(image_array)
        cv2.imshow(self.__window_name, image_array)
        if cv2.waitKey(1) & 0x7F == 27:
            print("Exit requested.")
            exit(0)

    def close(self):
        cv2.destroyWindow(self.__window_name)


if __name__ == '__main__':
    import time

    fs = OpenCVScreen(0)
    img = np.zeros((*fs.shape, 3), dtype=np.uint8)
    img[::2,::2,0] = 255
    # img[1::2,1::2,2] = 255
    # img[:,:,0] = 255
    # img[:,:,1] = 0
    # img[:,:,2] = 0
    # img[:] = 0
    img[:, [1,-2], 0] = 255
    img[[1,-2], :, 0] = 255
    rng = range(0, fs.shape[1], 20)
    time.sleep(1)
    start_time = time.time()
    for idx in rng:
        img[:, idx, 1] = 255
        fs.show(img)

    total_time = time.time() - start_time
    frames_per_second = len(rng) / total_time
    log.info("FullScreen display at {:0.1f} frames per second.".format(frames_per_second))


    fs.close()

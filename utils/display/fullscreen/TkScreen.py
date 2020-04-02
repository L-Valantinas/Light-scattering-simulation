from utils import log
from utils.display.fullscreen.Screen import Screen
import numpy as np
import tkinter as tk


class TkScreen(Screen):
    def __init__(self, screen_idx):
        super().__init__(screen_idx)

        # Set up a window and make it full-screen
        self.__window = tk.Tk(screenName="FullScreen{:d}".format(screen_idx), sync=True)
        self.__set_fullscreen(True)

        # def close_callback(event):
        #     self.close()
        #     return "break"
        # self.__window.bind("<Escape>", close_callback)

        # Prepare the bitmap
        image_array = np.zeros((self.screen.height, self.screen.width, 3), dtype=np.uint8)
        self.__raw_bytes_header = b'P6\n%i %i\n255\n' % (image_array.shape[1], image_array.shape[0])
        self.__image = tk.PhotoImage(data=self.__raw_bytes_header + image_array.tobytes())
        self.__canvas = tk.Canvas(self.__window, height=self.screen.height, width=self.screen.width,
                                  highlightbackground="#ff0000", highlightthickness=0)
        self.__canvas.create_image(self.screen.x, self.screen.y, anchor=tk.NW, image=self.__image)
        self.__canvas.pack()

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

        # Copy bytes to window on full screen display
        self.__image.put(self.__raw_bytes_header + image_array.tobytes())  # https://en.wikipedia.org/wiki/Netpbm_format
        self.__window.update()

    def __set_fullscreen(self, status=True):
        self.__window.attributes("-fullscreen", status)
        # self.window.attributes("-zoomed", status)   # Linux
        self.__window.attributes("-topmost", status)
        # self.__window.overrideredirect(True)

    def close(self):
        # self.__set_fullscreen(False)
        self.__window.destroy()


if __name__ == '__main__':
    import time

    fs = TkScreen(0)
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
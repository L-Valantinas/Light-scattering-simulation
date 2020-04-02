from abc import ABC, abstractmethod
import screeninfo


class Screen(ABC):
    def __init__(self, screen_idx):
        super().__init__()

        screen = type(self).get_available_screens()[screen_idx]
        self.__screen = screen
        self.__shape = (self.__screen.height, self.__screen.width)
        self.__callback = None

    @classmethod
    def get_available_screens(cls):
        # Detect the available screen devices
        available_screens = screeninfo.get_monitors()
        return available_screens

    @property
    def shape(self):
        return self.__shape

    @property
    def screen(self):
        return self.__screen

    @property
    def callback(self):
        return self.__callback

    @callback.setter
    def callback(self, new_callback):
        self.__callback = new_callback

    @abstractmethod
    def show(self, image_array):
        if self.callback is not None:
            self.callback(image_array)

    @abstractmethod
    def close(self):
        pass

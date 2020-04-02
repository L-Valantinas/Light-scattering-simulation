import numpy as np


def hsv2rgb(hsv_image):
    """
    Converts a hue-saturation-intensity value image to a red-green-blue image.

    :param hsv_image: a 3D numpy.array with the HSV image
    :return: rgb_image: a 3D numpy.array with the RGB image
    """
    # Convert HSV to an RGB image
    h = hsv_image[:, :, 0]
    s = hsv_image[:, :, 1]
    v = hsv_image[:, :, 2]

    h = 6.0 * h
    I = np.array(h, dtype=np.int8)
    f = h - I
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - (s * (1.0 - f)))

    I %= 6
    r = ((I == 0) | (I == 5)) * v + (I == 1) * q + ((I == 2) | (I == 3)) * p + (I == 4) * t
    g = (I == 0) * t + ((I == 1) | (I == 2)) * v + (I == 3) * q + (I >= 4) * p
    b = (I <= 1) * p + (I == 2) * t + ((I == 3) | (I == 4)) * v + (I == 5) * q
    rgb_image = np.concatenate((r[:, :, np.newaxis], g[:, :, np.newaxis], b[:, :, np.newaxis]), axis=2)

    return rgb_image

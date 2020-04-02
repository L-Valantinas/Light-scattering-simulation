import numpy as np
import utils.display


def complex2rgb(complex_image, normalization=None):
    """
    Converts a complex image to a RGB image.
    :param complex_image: A 2D array
    :param normalization: An optional scalar to indicate the target magnitude of the maximum value (1.0 is saturation).
    :return:
    """
    A = np.abs(complex_image)
    P = np.angle(complex_image)

    if normalization:
        if normalization > 0:
            max_value = np.max(abs(A.ravel()))
            if max_value > 0:
                A *= (normalization / max_value)

    V = np.minimum(A, 1.0)

    H = P / (2 * np.pi) + 0.5
    S = np.ones(H.shape)

    hsv_image = np.concatenate((H[:, :, np.newaxis], S[:, :, np.newaxis], V[:, :, np.newaxis]), axis=2)

    return utils.display.hsv2rgb(hsv_image)  # Convert HSV to an RGB image


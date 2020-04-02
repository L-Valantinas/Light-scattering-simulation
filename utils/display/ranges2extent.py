import numpy as np


def ranges2extent(*ranges):
    """
    Utility function to determine extent values for imshow
    :param ranges: monotonically increasing ranges, one per dimension (vertical, horizontal)
    :return: a 1D array
    """
    extent = []
    for idx, rng in enumerate(ranges[::-1]):
        rng = np.array(rng).ravel()
        step = rng[1] - rng[0]
        first, last = rng[0], rng[-1]
        if idx == 1:
            first, last = last, first
        extent.append(first - 0.5 * step)
        extent.append(last + 0.5 * step)

    return np.array(extent)



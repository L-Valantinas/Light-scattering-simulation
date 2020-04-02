import numpy as np


def calc_frequency_ranges(*ranges, centered=False):
    """
    Determine equivalent frequency ranges for given time ranges.
    The results are ifft-shifted so that the zero frequency is in the first
    vector position, unless centered=True.

    Examples:
       xfRange, yfRange = calc_frequency_ranges(xRange, yRange)
       xfRange, yfRange = calc_frequency_ranges(xRange, yRange, centered=True)

    :param ranges: one or more (spatial) time range vectors
    :param centered: Boolean indicating whether the resulting ranges should have the zero at the center. Default False.
    :return: one or more (spatial) frequency range vectors
    """
    f_ranges = []
    for rng in ranges:
        rng = rng.ravel()
        nb = rng.size
        if nb > 1:
            dt = np.array(rng[-1] - rng[0]) / (nb - 1)
            f_range = (np.arange(0, nb) - np.floor(nb / 2)) / (nb * dt)
        else:
            f_range = 0.0 * rng

        if not centered:
            f_range = np.fft.ifftshift(f_range)

        # Reshape to match input
        f_range.shape = rng.shape

        f_ranges.append(f_range)

    is_single_range = len(ranges) == 1
    if is_single_range:
        return f_ranges[0]  # No need to wrap it as a tuple
    else:
        return f_ranges

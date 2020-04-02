import numpy as np


def shift(arr: np.ndarray, nb_right_shifts: np.ndarray, axis=0, fill_value=0.0):
    """
    Shifts an nd-array along a given axis by nb_right_shifts and pads with the fill_value.
    The number of shifts can be negative and a vector of shifts can be specified, one for each dimension.

    :param arr: The array that will be copied and shifted.
    :param nb_right_shifts: The integer number of shifts to increased index value along axis. Or, a vector of integers
    for each axis. If a vector is specified, the axis-argument is ignored.
    :param axis: The axis to shift along.
    :param fill_value: The value to use for padding.

    :return: The shifted array, of identical type and shape as the input array, arr.
    """

    if not np.isscalar(nb_right_shifts):
        shifted_array = arr
        nb_right_shifts = np.array(nb_right_shifts).ravel()
        for ax in range(nb_right_shifts.size):
            shifted_array = shift(shifted_array, nb_right_shifts[ax], axis=ax, fill_value=fill_value)
    else:
        shifted_array = np.empty_like(arr)
        nb_right_shifts = int(nb_right_shifts)

        if axis != 0:
            arr = arr.swapaxes(0, axis)  # This should be a view in modern versions of numpy
            shifted_array = shifted_array.swapaxes(0, axis)

        if nb_right_shifts > 0:
            shifted_array[:nb_right_shifts, ...] = fill_value
            shifted_array[nb_right_shifts:, ...] = arr[:-nb_right_shifts, ...]
        elif nb_right_shifts < 0:
            shifted_array[nb_right_shifts:, ...] = fill_value
            shifted_array[:nb_right_shifts, ...] = arr[-nb_right_shifts:, ...]
        else:
            shifted_array[:] = arr

        if axis != 0:
            # swap axes back
            shifted_array = shifted_array.swapaxes(0, axis)

    return shifted_array


if __name__ == '__main__':
    a = np.arange(10,20,2)[:, np.newaxis] * np.arange(1, 4)[np.newaxis, :]
    print(a)
    print(shift(a, 2))
    print(shift(a, 1, axis=1))
    print(shift(a, -1))
    print(shift(a, (2, -1)))
    print(a)

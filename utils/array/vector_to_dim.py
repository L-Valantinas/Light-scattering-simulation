import numpy as np


def vector_to_dim(vec, n, axis=0):
    """
    Adds singleton dimensions to a 1D vector up to dimension n and orients the vector in dimension axis (default 0)
    :param vec: the input vector
    :param n: the number of desired dimensions
    :param axis: the target axis (default: 0)
    :return: a n-dimensional array with all-but-one singleton dimension
    """
    vec = np.array(vec, copy=True)
    indexes = [1]*n
    indexes[axis] = vec.shape[0]

    return vec.reshape(indexes)
import numpy as np
import utils.array


def calc_ranges(range_lengths, sample_pitches=[], center_offsets=[]):
    """
     calc_ranges(range_lengths, sample_pitches, center_offsets)

     Returns uniformly spaced ranges of length range_lengths(idx) with a elements spaced by
     sample_pitches[idx] and centered on center_offsets[idx]. The center element
     is defined as the one in the center for an odd number of elements and the
     next one for an even number of elements. If a scalar is specified as sample const.pitch
     or center offset, it is used for all ranges. The default sample const.pitch is 1
     and the default center offset is 0.
     The ranges are nd-arrays of a dimension equal to the maximum number of elements in range_lengths,
     sample_pitches, and center_offsets. Each range is a vector pointing along its respective dimension.

     :returns a tuple of ranges, one for each range_length.
     If range_lengths is scalar, a single range is returned, not a tuple of a range.

     Example:
      xRange = calc_ranges(128, 1e-6)
      xRange, yRange = calc_ranges(np.array([128, 128]), np.array([1, 1])*1e-6)

    """
    is_single_range = np.isscalar(range_lengths)
    # Make sure the vectors are of the same length
    nb_dims = np.max((np.array(range_lengths).size, np.array(sample_pitches).size, np.array(center_offsets).size))
    range_lengths = pad_to_length(range_lengths, nb_dims, 1)
    sample_pitches = pad_to_length(sample_pitches, nb_dims, 1)
    center_offsets = pad_to_length(center_offsets, nb_dims, 0)

    ranges = [co + sp * (np.arange(0, rl) - np.floor(rl / 2)) for co, sp, rl in
              zip(center_offsets, sample_pitches, range_lengths)]

    # Point each range in the right dimension and make it nd_dims-dimensional
    ranges = [utils.array.vector_to_dim(rng, nb_dims, axis=dim_idx) for (dim_idx, rng) in enumerate(ranges)]

    if is_single_range:
        return ranges[0]  # No need to wrap it as a tuple
    else:
        return ranges


def pad_to_length(vector, length, value=0):
    """
    Pads a 1D vector to a given length.

    :param vector: The input vector. Must be no longer than length.
    :param length: The length of the output vector.
    :param value: The value to use for padding.

    :return: An output vector of the specified length in which the first values coincide and the rest is filled up with the provided value.
    """
    vector = np.array(vector)
    result = value * np.ones(length, dtype=vector.dtype)
    result[:vector.size] = vector.ravel()

    return result

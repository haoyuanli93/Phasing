import numpy as np
import scipy as sp
import numba

"""
    Some functions are shared by both cpu and gpu algorithms,
    I'll just put them here.
    
"""


def shrink_wrap():
    pass


def check_algorithm_configuritions():
    pass


@numba.vectorize([numba.float32(numba.complex64), numba.float64(numba.complex128)])
def abs2(x):
    """
    Calculate the norm of the vector
    :param x:
    :return:
    """

    return x.real ** 2 + x.imag ** 2


def calculate_radial_distribution_simple(pattern, origin, bin_num=300):
    """
    This function will be used to provide values to fill those gaps in the detector when
    the user would like to use the auto-correlation as the initial support.

    :param pattern:
    :param origin:
    :param bin_num:
    :return:
    """
    dim = len(pattern.shape)
    shape = pattern.shape

    if dim != origin.shape[0]:
        raise Exception("The length of the origin array has to be the same as the dimension number"
                        "of the pattern array. i.e. len(pattern.shape)==origin.shape[0] ")

    # Calculate the distance regardless of the
    distance = sum(np.meshgrid(*[np.square(np.arange(shape[x]) - origin[x]) for x in range(dim)]))
    np.sqrt(distance, out=distance)

    catmap, ends = get_category_map(value_pattern=distance, bin_num=bin_num)
    mean_holder = np.zeros(bin_num, dtype=np.float64)
    for l in range(bin_num):
        mean_holder[l] = np.mean(pattern[catmap == l])

    return catmap, mean_holder, ends, distance


def get_category_map(value_pattern, bin_num):
    """
    This is a simple function to calculate a category map for the


    :param value_pattern:
    :param bin_num:
    :return:
    """

    value_bins = np.linspace(np.min(value_pattern) * 0.9, np.max(value_pattern) * 1.1,
                             num=bin_num + 1, endpoint=True)
    ends = np.zeros((bin_num, 2), dtype=np.float64)
    ends[:, 0] = value_bins[:-1]
    ends[:, 1] = value_bins[1:]

    category_map = np.ones_like(value_pattern, dtype=np.int64) * bin_num

    for l in range(bin_num):
        category_map[(value_pattern > ends[l, 0]) & (value_pattern <= ends[l, 1])] = l

    return category_map, ends

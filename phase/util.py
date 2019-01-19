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


def calculate_radial_distribution_simple(pattern, origin):
    """
    This function is used to fill those gaps in the detector when the user would like to
    use the auto-correlation as the support.

    :param pattern:
    :param origin:
    :return:
    """
    dim = len(pattern.shape)
    shape = pattern.shape

    if dim != origin.shape[0]:
        raise Exception("The length of the origin array has to be the same as the dimension number"
                        "of the pattern array. i.e. len(pattern.shape)==origin.shape[0] ")

    if dim == 3:
        grid_x, grid_y, grid_z = np.meshgrid(np.arange(shape[0]) - origin[0],
                                             np.arange(shape[1]) - origin[1],
                                             np.arange(shape[2]) - origin[2]
                                             )

        distance = np.sqrt(np.square(grid_x) + np.square(grid_y) + np.square(grid_z))

        distance_bins = np.linspace(np.min(distance) * 0.9, np.max(distance) * 1.1,
                                    num=300, endpoint=True)

        

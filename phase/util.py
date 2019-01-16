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

    return x.real**2 + x.imag**2

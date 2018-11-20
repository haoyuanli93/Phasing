import cmath
import math
from numba import cuda
import numpy as np


@cuda.jit('void(int64, int64, int64, float64, float64[:,:,:],' +
          ' boolean[:,:,:], float64[:,:,:], float64[:,:,:])')
def apply_support_constrain(shape_0, shape_1, shape_2, beta,
                            density_no_constrain, mask,
                            density_with_constrain,
                            previous_density):
    """
    Apply the HIO update for the patterns.

    :param shape_0: The shape [0] of the diffraction pattern to inspect
    :param shape_1: The shape [1] of the diffraction pattern to inspect
    :param shape_2: The shape [2] of the diffraction pattern to inspect
    :param beta: The update coefficient to use
    :param density_no_constrain: The derived density from the diffraction field without
                                    the support constrain
    :param mask: The mask to use. This is a boolean array.
    :param density_with_constrain: The derived density from the diffraction field with
                                    the support constrain
    :param previous_density: The previous estimation of the density function.
    :return: None
    """
    # Get grid index
    i, j, k = cuda.grid(3)

    # Make sure that the grid is not out of the pattern
    if i < shape_0 and j < shape_1 and k < shape_2:

        # Apply the constrain
        if mask[i, j, k] and (density_no_constrain[i, j, k] > 0):
            # Only in this situation,
            # one does not need to modify the ifft_pattern
            density_with_constrain[i, j, k] = density_no_constrain[i, j, k]
        else:
            density_with_constrain[i, j, k] = (previous_density[i, j, k] -
                                               beta * density_no_constrain[i, j, k])

        # Update the real space guess
        previous_density[i, j, k] = density_with_constrain[i, j, k]


@cuda.jit('void(int64, int64, int64, complex128[:,:,:],' +
          'complex128[:,:,:], complex128[:,:,:])')
def apply_magnitude_constrain(shape_0, shape_1, shape_2,
                              magnitude_constrain,
                              diffraction_no_constrain,
                              diffraction_with_constrain):
    """
    Apply the diffraction constrain

    :param shape_0: The shape [0] of the diffraction pattern to inspect
    :param shape_1: The shape [1] of the diffraction pattern to inspect
    :param shape_2: The shape [2] of the diffraction pattern of inspect
    :param magnitude_constrain: The diffraction pattern
    :param diffraction_no_constrain: The guessed diffraction pattern
    :param diffraction_with_constrain:  The modified guess of the diffraction
    :return: None
    """
    # Get grid index
    i, j, k = cuda.grid(3)

    # Make sure that the grid is not out of the pattern
    if i < shape_0 and j < shape_1 and k < shape_2:
        # Keep the phase from the guess and
        # apply the constrain from the diffraction pattern
        phase = cmath.phase(diffraction_no_constrain[i, j, k])
        diffraction_with_constrain[i, j, k] = (magnitude_constrain[i, j, k] *
                                               complex(math.cos(phase),
                                                       math.sin(phase)))


@cuda.jit('void(int64, int64, int64, complex128[:,:,:],' +
          'complex128[:,:,:], complex128[:,:,:], boolean[:,:,:])')
def apply_magnitude_constrain_with_mask(shape_0, shape_1, shape_2,
                                        magnitude_constrain,
                                        diffraction_no_constrain,
                                        diffraction_with_constrain,
                                        reciprocal_mask):
    """
    Apply the diffraction constrain with mask in the reciprocal space.

    :param shape_0:
    :param shape_1:
    :param shape_2:
    :param magnitude_constrain:
    :param diffraction_no_constrain:
    :param diffraction_with_constrain:
    :param reciprocal_mask:
    :return:
    """
    # Get grid index
    i, j, k = cuda.grid(3)

    # Make sure that the grid is not out of the pattern
    if i < shape_0 and j < shape_1 and k < shape_2:

        if reciprocal_mask[i, j, k]:
            # Keep the phase from the guess and
            # apply the constrain from the diffraction pattern
            diffraction_with_constrain[i, j, k] = (magnitude_constrain[i, j, k] *
                                                   cmath.exp(1j * cmath.phase(
                                                       diffraction_no_constrain[i, j, k])))


@cuda.jit('void(int64, int64, int64, float64[:,:,:], complex128[:,:,:])')
def get_real_part(shape_0, shape_1, shape_2, real_array, complex_array):
    """
    This function save the real part of the
     complex array into the real_part variable.

    :param shape_0:
    :param shape_1:
    :param shape_2:
    :param real_array:
    :param complex_array:
    :return:
    """
    # Get grid index
    i, j, k = cuda.grid(3)

    # Make sure that the grid is not out of the pattern
    if i < shape_0 and j < shape_1 and k < shape_2:
        real_array[i, j, k] = complex_array[i, j, k].real


@cuda.jit('void(int64, int64, int64, float64[:,:,:], complex128[:,:,:])')
def cast_to_complex(shape_0, shape_1, shape_2, real_array, complex_array):
    """
    This function store the real array into the complex array by setting the
    imaginary part to be zero uniformly

    :param shape_0:
    :param shape_1:
    :param shape_2:
    :param real_array:
    :param complex_array:
    :return:
    """
    # Get grid index
    i, j, k = cuda.grid(3)

    # Make sure that the grid is not out of the pattern
    if i < shape_0 and j < shape_1 and k < shape_2:
        complex_array[i, j, k] = np.complex128(real_array[i, j, k])


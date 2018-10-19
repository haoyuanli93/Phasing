import cmath

from numba import cuda


@cuda.jit('void(int64, int64, float64, float64[:,:], boolean[:,:], float64[:,:], float64[:,:])')
def apply_real_space_constrain_and_update_guess(shape_0, shape_1, beta, real_ifft_pattern, mask,
                                                real_constrain_ifft_pattern, guess_real_space):
    """
    Apply the HIO update for the patterns.

    :param shape_0: The shape [0] of the diffraction pattern to inspect
    :param shape_1: The shape [1] of the diffraction pattern to inspect
    :param beta: The update coefficient to use
    :param real_ifft_pattern: The real part of the ifft pattern
    :param mask: The mask to use. This is a boolean array.
    :param real_constrain_ifft_pattern: The ifft pattern modified by the real constrain
    :param guess_real_space: The previous guess
    :return: None
    """
    # Get grid index
    i, j = cuda.grid(2)

    # Make sure that the grid is not out of the pattern
    if i < shape_0 and j < shape_1:

        # Apply the constrain
        if mask[i, j] and (real_ifft_pattern[i, j] > 0):
            # Only in this situation, one does not need to modify the ifft_pattern
            real_constrain_ifft_pattern[i, j] = real_ifft_pattern[i, j]
        else:
            real_constrain_ifft_pattern[i, j] = guess_real_space[i, j] - beta * real_ifft_pattern[i, j]

        # Update the real space guess
        guess_real_space[i, j] = real_constrain_ifft_pattern[i, j]


@cuda.jit('void(int64, int64, complex128[:,:], complex128[:,:], complex128[:,:])')
def apply_diffraction_constrain(shape_0, shape_1, diffraction_constrain,
                                guess_diffraction, magnitude_constrain_pattern):
    """
    Apply the diffraction constrain

    :param shape_0: The shape [0] of the diffraction pattern to inspect
    :param shape_1: The shape [1] of the diffraction pattern to inspect
    :param diffraction_constrain: The diffraction pattern
    :param guess_diffraction: The guessed diffraction pattern
    :param magnitude_constrain_pattern:  The modified guess of the diffraction
    :return: None
    """
    # Get grid index
    i, j = cuda.grid(2)

    # Make sure that the grid is not out of the pattern
    if i < shape_0 and j < shape_1:
        # Keep the phase from the guess and apply the constrain from the diffraction pattern
        magnitude_constrain_pattern[i, j] = (diffraction_constrain[i, j] *
                                             cmath.exp(1j * cmath.phase(guess_diffraction[i, j])))


@cuda.jit('void(int64, int64, float64[:,:], complex128[:,:])')
def get_real_part(shape_0, shape_1, real_part, complex_array):
    # Get grid index
    i, j = cuda.grid(2)

    # Make sure that the grid is not out of the pattern
    if i < shape_0 and j < shape_1:
        real_part[i, j] = complex_array[i, j].real


@cuda.jit('void(int64, int64, float64[:,:], complex128[:,:])')
def cast_to_complex(shape_0, shape_1, real_part, complex_array):
    # Get grid index
    i, j = cuda.grid(2)

    # Make sure that the grid is not out of the pattern
    if i < shape_0 and j < shape_1:
        complex_array[i, j] = real_part[i, j] + 0j

from numba import cuda


@cuda.jit('void(int64, int64, float64, float64[:,:], bool[:,:], float64[:,:], float64[:,:])')
def apply_real_space_constrain(shape_0, shape_1, beta, real_ifft_pattern, mask, real_constrain_ifft_pattern,
                               previous_guess):
    """
    Apply the HIO update for the patterns.

    :param shape_0: The shape [0] of the diffraction pattern to inspect
    :param shape_1: The shape [1] of the diffraction pattern to inspect
    :param beta: The update coefficient to use
    :param real_ifft_pattern: The real part of the ifft pattern
    :param mask: The mask to use. This is a boolean array.
    :param real_constrain_ifft_pattern: The ifft pattern modified by the real constrain
    :param previous_guess: The previous guess
    :return: None
    """
    # Get grid index
    i, j = cuda.grid(2)

    # Make sure that the grid is not out of the pattern
    if i < shape_0 and j < shape_1:

        # Apply the constrain
        if mask[i, j] & real_ifft_pattern[i, j] > 0:
            # Only in this situation, one does not need to modify the ifft_pattern
            real_constrain_ifft_pattern[i, j] = real_ifft_pattern[i, j]
        else:
            real_constrain_ifft_pattern[i, j] = previous_guess[i, j] - beta * real_ifft_pattern[i, j]
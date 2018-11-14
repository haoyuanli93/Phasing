import cmath

from numba import cuda


@cuda.jit('void(int64, int64, float64, float64[:,:],' +
          ' boolean[:,:], float64[:,:], float64[:,:])')
def apply_real_space_constrain_and_update_guess(shape_0, shape_1, beta,
                                                real_ifft_pattern, mask,
                                                real_constrain_ifft_pattern,
                                                guess_real_space):
    """
    Apply the HIO update for the patterns.

    :param shape_0: The shape [0] of the diffraction pattern to inspect
    :param shape_1: The shape [1] of the diffraction pattern to inspect
    :param beta: The update coefficient to use
    :param real_ifft_pattern: The real part of the ifft pattern
    :param mask: The mask to use. This is a boolean array.
    :param real_constrain_ifft_pattern: The ifft pattern
                                        modified by the real constrain
    :param guess_real_space: The previous guess
    :return: None
    """
    # Get grid index
    i, j = cuda.grid(2)

    # Make sure that the grid is not out of the pattern
    if i < shape_0 and j < shape_1:

        # Apply the constrain
        if mask[i, j] and (real_ifft_pattern[i, j] > 0):
            # Only in this situation,
            # one does not need to modify the ifft_pattern
            real_constrain_ifft_pattern[i, j] = real_ifft_pattern[i, j]
        else:
            real_constrain_ifft_pattern[i, j] = (guess_real_space[i, j] -
                                                 beta * real_ifft_pattern[i, j])

        # Update the real space guess
        guess_real_space[i, j] = real_constrain_ifft_pattern[i, j]


@cuda.jit('void(int64, int64, complex128[:,:],' +
          'complex128[:,:], complex128[:,:])')
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
        # Keep the phase from the guess and
        # apply the constrain from the diffraction pattern
        magnitude_constrain_pattern[i, j] = (diffraction_constrain[i, j] *
                                             cmath.exp(1j * cmath.phase(
                                                 guess_diffraction[i, j])))


@cuda.jit('void(int64, int64, complex128[:,:],' +
          'complex128[:,:], complex128[:,:], boolean[:,:])')
def apply_diffraction_constrain_mask(shape_0, shape_1,
                                     diffraction_constrain,
                                     guess_diffraction,
                                     magnitude_constrain_pattern,
                                     reciprocal_mask):
    """
    Apply the diffraction constrain with mask in the reciprocal space.

    :param shape_0:
    :param shape_1:
    :param diffraction_constrain:
    :param guess_diffraction:
    :param magnitude_constrain_pattern:
    :param reciprocal_mask:
    :return:
    """
    # Get grid index
    i, j = cuda.grid(2)

    # Make sure that the grid is not out of the pattern
    if i < shape_0 and j < shape_1:

        if reciprocal_mask[i, j]:
            # Keep the phase from the guess and
            # apply the constrain from the diffraction pattern
            magnitude_constrain_pattern[i, j] = (diffraction_constrain[i, j] *
                                                 cmath.exp(1j * cmath.phase(
                                                     guess_diffraction[i, j])))


@cuda.jit('void(int64, int64, float64[:,:], complex128[:,:])')
def get_real_part(shape_0, shape_1, real_part, complex_array):
    """
    This function save the real part of the
     complex array into the real_part variable.

    :param shape_0:
    :param shape_1:
    :param real_part:
    :param complex_array:
    :return:
    """
    # Get grid index
    i, j = cuda.grid(2)

    # Make sure that the grid is not out of the pattern
    if i < shape_0 and j < shape_1:
        real_part[i, j] = complex_array[i, j].real


@cuda.jit('void(int64, int64, float64[:,:], complex128[:,:])')
def cast_to_complex(shape_0, shape_1, real_part, complex_array):
    """
    This function store the real array into the complex array by setting the
    imaginary part to be zero uniformly

    :param shape_0:
    :param shape_1:
    :param real_part:
    :param complex_array:
    :return:
    """
    # Get grid index
    i, j = cuda.grid(2)

    # Make sure that the grid is not out of the pattern
    if i < shape_0 and j < shape_1:
        complex_array[i, j] = real_part[i, j] + 0j


@cuda.jit('void(int64, int64, int64, float64[:,:], float64[:,:], float64[:,:])')
def apply_filter(f_range, filter_start,
                 filter_end, filter_array,
                 raw_data, new_data):
    """
    This function apply the filter to the raw pattern and saves the processed
    pattern to the new_data variable.

    :param f_range: Assume that the shape of the filter array is [2n+1, 2n+1]
                    Then f_range = n
    :param filter_start: the first index to process
    :param filter_end: the last index to process
    :param filter_array: The 2D array containing the filter
    :param raw_data: The raw pattern array to process
    :param new_data: The variable to hold the processed array.
    :return:
    """
    i, j = cuda.grid(2)

    if filter_start < i < filter_end and filter_start < j < filter_end:

        # initialize the value
        new_data[i, j] = 0

        # for loop through the filter to get the value.
        for l in range(-f_range, f_range):
            for m in range(-f_range, f_range):
                new_data[i, j] += (filter_array[l + f_range, m + f_range] *
                                   raw_data[i + l, j + m])


@cuda.jit('void(int64, int64, float64[:,:], float64, boolean[:,:])')
def take_threshold(shape_0, shape_1, raw_data, threshold, new_data):
    """
    The difference map needs to use a threshold to generate a new support.

    :param shape_0:
    :param shape_1:
    :param raw_data:
    :param threshold:
    :param new_data:
    :return:
    """
    i, j = cuda.grid(2)

    if i < shape_0 and j < shape_1:
        if raw_data[i, j] > threshold:
            new_data[i, j] = True


@cuda.jit('void(int64, int64, boolean[:,:], boolean[:,:], boolean[:,:])')
def combine_two_supports(shape_0, shape_1,
                         input_support_1, input_support_2, output_support):
    """
    Sometimes, the user might know something about the support.
    This function do an element-wise and on the two input boolean array
    The result is saved to the output_support

    :param shape_0:
    :param shape_1:
    :param input_support_1:
    :param input_support_2:
    :param output_support:
    :return:
    """
    i, j = cuda.grid(2)

    if i < shape_0 and j < shape_1:
        output_support[i, j] = (input_support_1[i, j] and input_support_2[i, j])

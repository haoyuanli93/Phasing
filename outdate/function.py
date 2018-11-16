
@cuda.jit(str('void(int64, int64[:], int64[:],' +
              'float64[:,:], float64[:,:], float64[:,:])'))
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

    if (filter_start[0] < i < filter_end[0]
            and filter_start[1] < j < filter_end[1]):

        # initialize the value
        new_data[i, j] = 0

        # for loop through the filter to get the value.
        for l in range(-f_range, f_range + 1):
            for m in range(-f_range, f_range + 1):
                new_data[i, j] += (filter_array[l + f_range, m + f_range] *
                                   raw_data[i + l, j + m])


@cuda.jit('void(int64, int64, float64[:,:], float64[:], float64, boolean[:,:])')
def take_threshold(shape_0, shape_1, raw_data, max_value, threshold_ratio,
                   new_data):
    """
    The difference map needs to use a threshold to generate a new support.

    :param shape_0:
    :param shape_1:
    :param raw_data:
    :param max_value:
    :param threshold_ratio:
    :param new_data:
    :return:
    """
    i, j = cuda.grid(2)

    threshold = max_value[0] * threshold_ratio

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


@cuda.jit('void(float64[:], float64[:,:])')
def get_max_2d(result, values):
    """
    Get the max value from the array

    :param result:
    :param values:
    :return:
    """
    i, j = cuda.grid(2)
    # Atomically store to result[0] from values[i, j]
    cuda.atomic.max(result, 0, values[i, j])


@cuda.jit('void(float64[:], float64[:,:,:])')
def get_max_3d(result, values):
    """
    Get the max value from the array

    :param result:
    :param values:
    :return:
    """
    i, j, k = cuda.grid(3)
    # Atomically store to result[0] from values[i, j]
    cuda.atomic.max(result, 0, values[i, j, k])

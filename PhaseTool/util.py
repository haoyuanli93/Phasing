import numpy as np
from scipy import ndimage
from skimage import morphology
import numba

"""
    Some functions are shared by both cpu and gpu algorithms,
    I'll just put them here.
    
"""


def shrink_wrap(density, sigma=1, threshold_ratio=0.04, filling_holds=False, convex_hull=False):
    """
    This function derive

    :param density:
    :param convex_hull:
    :param sigma:
    :param threshold_ratio:
    :param filling_holds:
    :return:
    """
    density_smooth = ndimage.gaussian_filter(input=density, sigma=sigma)
    den_min = np.min(density_smooth)
    den_span = np.max(density_smooth) - den_min

    # Get a holder for the support
    support_tmp = np.zeros_like(density, dtype=np.bool)
    support = np.copy(support_tmp)

    # Take a threshold
    threshold = threshold_ratio * den_span + den_min

    # Apply the threshold
    support_tmp[density_smooth >= threshold] = True

    # Check if additional conditions are available.
    if filling_holds:
        print("As per request, fill holes in the support. The convex_hull argument is ignored.")
        ndimage.binary_fill_holes(input=support_tmp, output=support)

    elif convex_hull:
        print("As per request, use the convex hull of standard shrink-wrap result as the support.")
        support = morphology.convex_hull_image(support_tmp)
    else:
        print("Using the result of the standard shrink-wrap as the new support array.")
        support = np.copy(support_tmp)

    return support


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


def get_radial_info(pattern, pattern_mask, origin, bin_num=300):
    """
    This function will be used to provide values to fill those gaps in the detector when
    the user would like to use the auto-correlation as the initial support.

    :param pattern:
    :param pattern_mask
    :param origin:
    :param bin_num:
    :return:
    """
    dim = len(pattern.shape)
    shape = pattern.shape

    if dim != origin.shape[0]:
        raise Exception("The length of the origin array has to be the same as the dimension number"
                        "of the pattern array. i.e. len(pattern.shape)==origin.shape[0]")

    # Calculate the distance regardless of the
    distance = sum(np.meshgrid(*[np.square(np.arange(shape[x]) - origin[x]) for x in range(dim)]))
    np.sqrt(distance, out=distance)

    catmap, ends = get_category_map(value_pattern=distance, bin_num=bin_num)

    # Add the mask info to the catmap variable. Because one would not want to calculate
    # The average including the gaps.
    catmap_masked = np.copy(catmap)
    catmap_masked[np.logical_not(pattern_mask)] = bin_num

    # Get the region where there are some valid pixels.
    cat_start = np.min(catmap_masked)
    cat_stop = np.max(catmap_masked)

    # Get the mean value
    mean_holder = np.zeros(bin_num, dtype=np.float64)
    for l in range(cat_start, cat_stop):
        mean_holder[l] = np.mean(pattern[catmap == l])

    # Set the lower region values to the same as the
    mean_holder[:cat_start] = mean_holder[cat_start]

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


def get_support_from_autocorrelation(magnitude, magnitude_mask, origin,
                                     threshold=0.04,
                                     gaussian_filter=True,
                                     gaussian_sigma=1.,
                                     flag_fill_detector_gap=False,
                                     bin_num=300):
    """

    :param magnitude:
    :param magnitude_mask:
    :param origin:
    :param threshold:
    :param gaussian_filter:
    :param gaussian_sigma:
    :param flag_fill_detector_gap:
    :param bin_num:
    :return:
    """

    # Step 1. Check if I need to fix those gaps.
    if flag_fill_detector_gap:
        # Create a variable to handle both case at the same time
        data_tmp = fill_detector_gap(magnitude=magnitude,
                                     magnitude_mask=magnitude_mask,
                                     origin=origin,
                                     gaussian_filter=gaussian_filter,
                                     gaussian_sigma=gaussian_sigma,
                                     bin_num=bin_num)
    else:
        data_tmp = magnitude

    # Step 2. Get the correlation
    autocorrelation = np.fft.ifftn(np.square(data_tmp)).real

    if gaussian_filter:
        ndimage.gaussian_filter(input=autocorrelation,
                                sigma=gaussian_sigma,
                                output=autocorrelation)

    # Step 3. Get the threshold and get the support.

    # Set all the pixels with values lower than the threshold to be zero.
    # Notice that here, the ture threshold (threshold_t) is low + threshold * span
    low = np.min(autocorrelation)
    span = np.max(autocorrelation) - low

    threshold_t = low + threshold * span
    support_holder = np.ones_like(magnitude_mask, dtype=np.bool)
    support_holder[autocorrelation <= threshold_t] = False

    return support_holder


def fill_detector_gap(magnitude, magnitude_mask, origin, gaussian_filter=True,
                      gaussian_sigma=1., bin_num=300):
    """
    Fill the gaps in the detector will the corresponding average value for that radial region.

    :param magnitude:
    :param magnitude_mask:
    :param origin:
    :param gaussian_filter:
    :param gaussian_sigma:
    :param bin_num:
    :return:
    """

    # Get the radial info
    (catmap,
     mean_holder,
     ends,
     distance) = get_radial_info(pattern=magnitude,
                                 pattern_mask=magnitude_mask,
                                 origin=origin,
                                 bin_num=bin_num)

    # Fill the gaps
    magnitude_filled = np.zeros_like(magnitude)

    # Create a tmp mask for convenience
    mask_tmp = np.zeros_like(magnitude, dtype=np.bool)
    magmask_tmp = np.logical_not(magnitude_mask)

    for l in range(bin_num):
        mask_tmp[:] = False
        mask_tmp[(catmap == bin_num) & magmask_tmp] = True

        magnitude_filled[mask_tmp] = mean_holder[l]

    if gaussian_filter:
        magnitude_filled = ndimage.gaussian_filter(input=magnitude_filled,
                                                   sigma=gaussian_sigma)

    return magnitude_filled


def approximate_magnitude_projection(dens, mag, epsilon):
    """
       This is a new operator to replace the original magnitude operator. According to professor
    Luke in the paper

        http://iopscience.iop.org/article/10.1088/0266-5611/21/1/004

        Relaxed averaged alternating reflections for diffraction imaging

    This new operator is more stable. Concrete formula is (34) in the paper. Notice that the
    formula contains a typo. I here represents the identity operator and should be replaced by u.


    :param dens: The old updated density.
    :param mag: The magnitude array. Notice that, here, the magnitude array might not be the
                original one. Because the original array have edges if one simply assign zeros to
                the missing data, one might consider to assign values from the estimation to reduce
                the artifical edges.
    :param epsilon: A epsilon value used to calculate the true epsilon value. The detail should be
                    find from the article.
    :return:
    """
    # Get the fourier transform
    holder_1 = np.fft.fftn(dens)

    # Get the norm of the transformed data
    holder_2 = abs2(holder_1)

    # Calculate the true epsilon that should be used in the calculation
    teps = (epsilon * np.max(holder_2)) ** 2

    # Calculatet the output without truely return any array
    holder_3 = np.divide(np.multiply(holder_2 - np.multiply(mag, np.sqrt(holder_2 + teps)),
                                     np.multiply(holder_2 + 2 * teps, holder_1)),
                         np.square(holder_2 + teps))
    return dens - np.fft.ifftn(holder_3)




####################################################################################################
#  For test
####################################################################################################
@numba.jit
def create_disk(space, center, radius):

    for l in range(center[0] - radius, center[0] + radius):
        for m in range(center[1] - radius, center[1] + radius):
            space[l, m] = 
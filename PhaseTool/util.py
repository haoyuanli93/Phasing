import numpy as np
from scipy import ndimage
from scipy import stats
from skimage import morphology
import numba
from numba import float64, int64, void

"""
    Some functions are shared by both cpu and gpu algorithms,
    I'll just put them here.
    
"""

epsilon = np.finfo(np.float64).eps


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
        ndimage.binary_fill_holes(input=support_tmp, output=support)

    elif convex_hull:
        support = morphology.convex_hull_image(support_tmp)
    else:
        support = np.copy(support_tmp)

    return support


def get_autocorrelation(image):
    """
    Get the autocorrelation of the image
    :param image:
    :return:
    """
    return np.fft.ifftshift(np.abs(np.fft.ifftn(image)))


def shift_to_center(array):
    """
    Shift the image so that the center of mass of this image is the same as the geometric center
    of the image.

    :param array:
    :return: The shifted array
    """
    # Get array information
    shape = array.shape
    dim = len(shape)
    center = np.array([x / 2.0 for x in shape])

    # Get the position of the center of mass
    center_mass = ndimage.center_of_mass(array)

    # Shift the pattern
    shifted_array = ndimage.shift(array,
                                  shift=[center[l] - center_mass[l] for l in range(dim)])

    return shifted_array


def resolve_trivial_ambiguity(array, reference_array):
    """
    Usually, the user would have to do multiple recovers to finally determine the phase.
    However, for different reconstructions from the same magnitude, there could be some
    trivial ambiguity. This functions aims to resolve this problem.

    The way to resolve this ambiguity is to compare with a reference array.

    The detail should be obvious if you read the source code.

    :param array:
    :param reference_array:
    :return: The transformed array.
    """

    ################################################################################################
    # Step 1, Shift the reference array to the center
    ################################################################################################
    # Shifted reference array
    sr_array = shift_to_center(reference_array)

    ################################################################################################
    # Step 2, Calculate the correlation between the flipped and shifted array with the sr_array
    ################################################################################################

    # Get the dimension number
    dim = len(array.shape)

    # Generate a list of all possible flips
    tmp_list = np.meshgrid(*(np.array([0, 1], dtype=np.int64),) * dim)
    flip_list = np.transpose(np.stack((x.flatten() for x in tmp_list)))
    flip_num = np.power(2, dim)

    # Loop through all the possible flips
    correlation = np.zeros(flip_num, dtype=np.float64)  # Holder for correlation value
    for l in range(flip_num):

        flip_array = np.copy(array)  # Holder for the flipped array
        # Calculated the flipped array
        for m in range(dim):
            if flip_list[l, m] == 1:
                flip_array = np.flip(flip_array, axis=m)

        # Calculate the shifted array
        shifted_array = shift_to_center(flip_array)

        # Calculate the correlation between the reference and the shifted array
        correlation[l] = stats.pearsonr(shifted_array, sr_array)[0]

    # Find the array with the largest correlation
    idx = np.argmax(correlation)
    cor_val = correlation[idx]

    # Constructed the selected array
    flip_array = np.copy(array)
    for m in range(dim):
        if flip_list[idx, m] == 1:
            flip_array = np.flip(flip_array, axis=m)

    shifted_array = shift_to_center(flip_array)

    return shifted_array, sr_array, cor_val


@numba.vectorize([numba.float32(numba.complex64), numba.float64(numba.complex128)])
def abs2(x):
    """
    Calculate the norm of the vector
    :param x:
    :return:
    """

    return x.real ** 2 + x.imag ** 2


@numba.vectorize([numba.complex64(numba.complex64), numba.complex128(numba.complex128)])
def get_phase(x):
    """
    Calculate the norm of the vector
    :param x:
    :return:
    """
    value = np.abs(x)
    if value >= epsilon:
        return x / value
    return x


def get_beam_stop_mask(space_shape, space_center, radius):
    """
    Get a boolean mask about the beam stop.

    :param space_shape: The shape of the space
    :param space_center: The center of the space
    :param radius: The radius of the beam stop
    :return:
    """

    dim = len(space_shape)

    # Calculate the distance
    distance = sum(np.meshgrid(*[np.square(np.arange(space_shape[x], dtype=np.float64)
                                           - float(space_center[x])) for x in range(dim)]))
    np.sqrt(distance, out=distance)

    # Holder for the mask
    mask = np.ones(space_shape, dtype=np.bool)
    mask[distance <= radius] = False

    mask_not = np.logical_not(mask)

    return mask, mask_not


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
    autocorrelation = np.abs(np.fft.ifftn(np.square(data_tmp)))

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


def approximate_magnitude_projection(dens, mag, eps):
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
    :param eps: A epsilon value used to calculate the true epsilon value. The detail should be
                    find from the article.
    :return:
    """
    # Get the fourier transform
    holder_1 = np.fft.fftn(dens)

    # Get the norm of the transformed data
    holder_2 = abs2(holder_1)

    # Calculate the true epsilon that should be used in the calculation
    teps = (eps * np.max(holder_2)) ** 2

    # Calculatet the output without truely return any array
    holder_3 = np.divide(np.multiply(holder_2 - np.multiply(mag, np.sqrt(holder_2 + teps)),
                                     np.multiply(holder_2 + 2 * teps, holder_1)),
                         np.square(holder_2 + teps))
    return dens - np.fft.ifftn(holder_3)


####################################################################################################
#  For test
####################################################################################################
@numba.jit(void(float64[:, :], int64[:], int64, int64), nopython=True, parallel=True)
def create_disk(space, center, radius, radius_square):
    """
    This function create a disk.

    :param space:
    :param center:
    :param radius:
    :param radius_square:
    :return:
    """
    for l in range(- radius, radius):
        for m in range(- radius, radius):
            if l * l + m * m <= radius_square:
                space[l + center[0], m + center[1]] += 1.


@numba.jit(void(float64[:, :, :], int64[:], int64, int64), nopython=True, parallel=True)
def create_sphere(space, center, radius, radius_square):
    """
    This function create a disk.

    :param space:
    :param center:
    :param radius:
    :param radius_square:
    :return:
    """
    for l in range(- radius, radius):
        for m in range(- radius, radius):
            for n in range(-radius, radius):
                if l * l + m * m <= radius_square:
                    space[l + center[0], m + center[1], n + center[2]] += 1.


def get_smooth_sample(dim=2, space_length=128, support_length=48,
                      obj_num=50, rlim_low=1,
                      rlim_high=6):
    """
    This function returns a smooth sample for test.

    :param dim: The dimension of the sample space
    :param space_length:
    :param support_length:
    :param obj_num:
    :param rlim_low: This function generates spheres or disks as examples. rlim_low represent the
                    lowest radius value for the random sphere radius.
    :param rlim_high: The hightest radius value.
    :return:
    """

    # Step 1: Get the center of the space and change the format.
    obj_num = int(obj_num)
    center = (int(space_length / 2.),) * dim

    # Step 2: Generate 50 random center position and 50 random length
    center_list = np.random.randint(low=center[0] - int(support_length / 2.0),
                                    high=center[0] + int(support_length / 2.0),
                                    size=(obj_num, dim),
                                    dtype=np.int64)
    radius_list = np.random.randint(low=rlim_low, high=rlim_high, size=obj_num)
    radius_square = np.square(radius_list)

    # Step 3: Use the create_disk function to create these objects in the space
    space = np.zeros((space_length,) * dim, dtype=np.float64)

    if dim == 2:
        for l in range(obj_num):
            create_disk(space=space,
                        center=center_list[l],
                        radius=radius_list[l],
                        radius_square=radius_square[l])
    elif dim == 3:
        for l in range(obj_num):
            create_sphere(space=space,
                          center=center_list[l],
                          radius=radius_list[l],
                          radius_square=radius_square[l])

    else:
        raise Exception("At present, this function can only handle 2 and 3 d case. "
                        "Therefore, the argument dim has to be 2 or 3.")

    ndimage.gaussian_filter(input=space, sigma=2, output=space)

    return space


class SmoothSample:
    def __init__(self, dim=2, space_length=128, support_length=48, total_density=1e4,
                 beam_stop=2, shot_noise=True, gaussian=True,
                 gaussian_mean=0, gaussian_sigma=1,
                 obj_num=50, rlim_low=1, rlim_high=6):
        """
        This generate a smooth sample and generate the corresponding diffraction information

        :param dim:
        :param total_density: The summation of the density. This controls the diffraction
                                strength.
        :param space_length: The length of the sample
        :param support_length: The region in which the centers of the objects are in
        :param beam_stop: The size of the beam stop
        :param shot_noise: Whether to apply the po
        :param gaussian:
        :param gaussian_mean:
        :param gaussian_sigma:
        :param obj_num: Number of disks or spheres in the sample
        :param rlim_low: This function generates spheres or disks as examples.
                    rlim_low represent the lowest radius value for the random sphere radius.
        :param rlim_high: The hightest radius value.
        """

        self.dim = dim
        self.center = np.array((space_length / 2.0,) * dim, dtype=np.float64)
        self.beam_stop = 0

        # The sample
        self.density = get_smooth_sample(dim=dim,
                                         space_length=space_length,
                                         support_length=support_length,
                                         obj_num=obj_num,
                                         rlim_high=rlim_high,
                                         rlim_low=rlim_low)

        # Normalize the sample
        tmp = np.sum(self.density, dtype=np.float64)
        tmp = float(total_density / tmp)
        self.density *= tmp

        # Things that can not be measured
        self.diffraction = np.fft.fftshift(np.fft.fftn(self.density))
        self.complex_phase = get_phase(self.diffraction)

        # Things that can be measured
        self.intensity = abs2(self.diffraction)
        self.magnitude = np.sqrt(self.intensity)

        # Things that can be derived
        self.autocorrelation = get_autocorrelation(self.intensity)

        # Consider various detector effect
        # 1. Beam stop
        self.detector_mask = np.ones_like(self.density)
        self.detector_mask_not = np.logical_not(self.detector_mask)

        self.det_intensity = np.copy(self.intensity)
        self.det_intensity[self.detector_mask_not] = 0.

        self.det_magnitude = np.sqrt(self.det_intensity)
        self.det_autocorrelation = get_autocorrelation(self.det_intensity)

        # 2. Detector Noise
        self.shot_noise_noise_flag = False
        self.gaussian_noise_flag = False
        self.gaussian_mean = 0
        self.gaussian_sigma = 1

        self.det_noisy_intensity = np.copy(self.intensity)
        self.det_noisy_intensity[self.detector_mask_not] = 0.

        self.det_noisy_magnitude = np.sqrt(self.det_intensity)
        self.det_noisy_autocorrelation = get_autocorrelation(self.det_intensity)

        # Apply the detector effects
        self._set_and_add_detector_effect(beam_stop=beam_stop,
                                          shot_noise=shot_noise,
                                          gaussian=gaussian,
                                          gaussian_mean=gaussian_mean,
                                          gaussian_sigma=gaussian_sigma)

    def estimate_support(self, threshold):
        """

        :param threshold:
        :return:  support[self.density >= threshold] = true
        """
        support = np.zeros_like(self.density, dtype=np.bool)
        support[self.density >= threshold] = True

        return support

    def _set_and_add_detector_effect(self, beam_stop,
                                     shot_noise=False,
                                     gaussian=False,
                                     gaussian_mean=0,
                                     gaussian_sigma=1):
        """
        Add detector noise.
        
        :param beam_stop:
        :param shot_noise: 
        :param gaussian: 
        :param gaussian_mean: 
        :param gaussian_sigma: 
        :return: 
        """
        self.beam_stop = beam_stop

        # Step 1: Create the mask
        (self.detector_mask,
         self.detector_mask_not) = get_beam_stop_mask(space_shape=self.density.shape,
                                                      space_center=self.center,
                                                      radius=self.beam_stop)
        # Step 2: Apply this mask
        self.det_intensity = np.copy(self.intensity)
        self.det_intensity[self.detector_mask_not] = 0.

        self.det_magnitude = np.sqrt(self.det_intensity)
        self.det_autocorrelation = get_autocorrelation(self.det_intensity)

        # Step 3: Add noise
        self.shot_noise_noise_flag = shot_noise
        self.gaussian_noise_flag = gaussian
        self.gaussian_sigma = gaussian_sigma
        self.gaussian_mean = gaussian_mean

        self.det_noisy_intensity = np.copy(self.det_intensity)
        if self.shot_noise_noise_flag:
            self.det_noisy_intensity = np.random.poisson(self.det_noisy_intensity)
            self.det_noisy_intensity = self.det_noisy_intensity.astype(np.float64)

        if self.gaussian_noise_flag:
            self.det_noisy_intensity += np.random.normal(self.gaussian_mean,
                                                         self.gaussian_sigma,
                                                         size=self.density.shape)

        # Fix negative values and values outside the beam stop mask
        self.det_noisy_intensity[self.det_noisy_intensity <= 0] = 0
        self.det_noisy_intensity[self.detector_mask_not] = 0

        # Calculate the magnitude and the
        self.det_noisy_magnitude = np.sqrt(self.det_noisy_intensity)
        self.det_noisy_autocorrelation = get_autocorrelation(self.det_noisy_intensity)

import numpy as np


def get_gaussian_filter_slow(extend, sigma, dimension):
    """
    Get a gaussian filter with length of 2 * extend + 1 along each dimension
    :param extend:
    :param sigma:
    :param dimension: Specify whether it's 2D or 3D
    :return:
    """

    if dimension == 2:

        holder = np.zeros((2 * extend + 1, 2 * extend + 1), dtype=np.float64)
        for l in range(-extend, extend + 1):
            for m in range(-extend, extend + 1):
                holder[l, m] = np.exp(-(l ** 2 + m ** 2) / (2 * sigma ** 2))

        # Normalize the filter
        holder /= np.sum(holder)

    elif dimension == 3:
        holder = np.zeros((2 * extend + 1,
                           2 * extend + 1,
                           2 * extend + 1), dtype=np.float64)
        for l in range(-extend, extend + 1):
            for m in range(-extend, extend + 1):
                for n in range(-extend, extend + 1):
                    holder[l, m] = np.exp(-(l ** 2 + m ** 2) / (2 * sigma ** 2))

        # Normalize the filter
        holder /= np.sum(holder)

    else:
        raise Exception("dimension has to be integer 2 or 3.")

    return holder


def get_gaussian_filter(extend, sigma, dimension):
    """
    Get a gaussian filter quickly.

    :param extend:
    :param sigma:
    :param dimension:
    :return:
    """
    coor = np.meshgrid(*tuple((np.arange(-extend,
                                         extend + 1,
                                         dtype=np.float64),) * dimension))

    holder = np.square(coor[0])
    for l in range(1, dimension):
        holder += np.square(coor[l])

    # Scale according to the sigma
    holder /= -(sigma ** 2)

    # Apply the exp function
    np.exp(holder, out=holder)

    return holder


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


l

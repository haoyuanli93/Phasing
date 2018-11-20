import numpy as np
import time
import scipy.ndimage as sn


def apply_hio_with_wrap_shrink(magnitude_constrain,
                               support_bool,
                               reciprocal_mask,
                               beta=0.8,
                               iter_num=1000,
                               threshold_ratio=0.04,
                               sigma_start=5,
                               sigma_stop=0.5,
                               support_decay_rate=50):
    """
    This function calculate the retrieved phase and the corresponding real space electron density
    in n-dimension.

    :param magnitude_constrain: This is the magnitude measured by the detector. This has to be
                                a 2d numpy array. It can be either np.float or np.complex since I
                                will convert it into complex variable at the beginning anyway.
    :param support_bool: This is the initial estimate of the support. This is a 2D boolean array.
    :param reciprocal_mask: This is the mask for the 2d detector. This algorithm only consider
                            values in the magnitude constrain with the corresponding values in this
                            mask being True.
    :param beta: The update rate in the HIO algorithm
    :param iter_num: The iteration number to apply the hio procedure
    :param threshold_ratio: The threshold ratio to update the support.
                            When constructing new supports from current estimations, this program
                            assume that the threshold is at
                                    min_value + threshold*(max_value - min_value)
    :param sigma_start: The largest sigma in the gaussian filter used to
                        smooth out the estimation of the density.
    :param sigma_stop: The smallest sigma in the gaussian filter used to
                        smooth out the estimation of the density.
    :param support_decay_rate: The number of iterations before the program update the support.
                                Notice that in this program, the sigma used to smooth out the
                                estimation of the density is calculated as a linear equation
                                sigma_list = np.linspace(sigma_start,
                                                        sigma_stop,
                                                         num=sigma_num, endpoint=True)
    :return:
    """
    tic = time.time()

    ###############################################################################################
    # Step 0: Generated required meta-parameters
    ###############################################################################################
    support_decay_rate = int(support_decay_rate)

    sigma_num = int(iter_num / support_decay_rate) + 1
    sigma_list = np.linspace(sigma_start, sigma_stop, num=sigma_num, endpoint=True)

    # Because later, I would need the opposite of the support, therefore I flip it here.
    flipped_support_bool = np.logical_not(support_bool)

    # Make a copy of the initial support
    initial_support = np.copy(support_bool)

    ###############################################################################################
    # Step 0: Create variables for calculation
    ###############################################################################################
    magnitude_constrain = np.ascontiguousarray(magnitude_constrain.astype(np.complex128))

    # Variable containing the diffraction satisfies the magnitude constrain
    phase_tmp = np.exp(1j * np.random.rand(*magnitude_constrain.shape) * 2 * np.pi)
    diffract_with_magnitude_constrain = np.ascontiguousarray(np.multiply(phase_tmp,
                                                                         magnitude_constrain))

    # Containing the previous result of the density function with support constrain
    density_previous = np.fft.ifftn(diffract_with_magnitude_constrain)
    density_previous = density_previous.real

    ###############################################################################################
    # Step 3: Begin calculation
    ###############################################################################################
    # Begin the loop
    for idx in range(iter_num):

        # Derive the density from the diffraction
        density_present = np.fft.ifftn(diffract_with_magnitude_constrain)

        # Get the real part
        density_present = np.real(density_present)

        # apply real space constraints
        density_present[(density_present < 0) |
                        flipped_support_bool] = (
                density_previous[(density_present < 0) |
                                 flipped_support_bool] - beta * density_present[
                    (density_present < 0) |
                    flipped_support_bool])

        # Update the previous estimation of the density for next iteration's calculation
        density_previous = np.copy(density_present)

        # Update the guess for the diffraction
        diffract_no_magnitude_constrain = np.fft.fftn(density_present)

        # apply fourier domain constraints
        phase_tmp = np.exp(1j * np.angle(diffract_no_magnitude_constrain))
        diffract_with_magnitude_constrain[reciprocal_mask] = (magnitude_constrain[reciprocal_mask] *
                                                              phase_tmp[reciprocal_mask])

        # Every 50 iterations, update the estimation of the support
        if np.mod(idx + 1, support_decay_rate) == 0:
            filtered = sn.filters.gaussian_filter(
                input=density_present,
                sigma=sigma_list[int((idx + 1) / support_decay_rate)])

            # Find the largest number of pixels
            max_value = np.max(filtered)
            min_value = np.min(filtered)

            flipped_support_bool = np.ones_like(magnitude_constrain, dtype=np.bool)
            flipped_support_bool[
                filtered > (min_value + threshold_ratio * (max_value - min_value))] = False

    toc = time.time()
    print("It takes {:.2f} seconds to do {} iterations.".format(toc - tic, iter_num))

    ###############################################################################################
    # Step 4: Put results into a dictionary
    ###############################################################################################
    result = {'Magnitude': magnitude_constrain,
              'Initial Support': initial_support,
              'Final Support': np.logical_not(flipped_support_bool),
              'Reconstructed Density': density_present,
              'Reconstructed Diffraction Field': diffract_no_magnitude_constrain,
              'Reconstructed Magnitude Field': np.abs(diffract_no_magnitude_constrain),
              'Calculation Time (s)': toc - tic,
              'Iteration number': iter_num,
              'Sigma list': sigma_list}

    return result


def apply_hio_no_wrap_shrink(magnitude_constrain,
                             support_bool,
                             reciprocal_mask,
                             beta=0.8,
                             iter_num=1000):
    """
    This function calculate the retrieved phase and the corresponding real space electron density
    in n-dimension.

    :param magnitude_constrain: This is the magnitude measured by the detector. This has to be
                                a 2d numpy array. It can be either np.float or np.complex since I
                                will convert it into complex variable at the beginning anyway.
    :param support_bool: This is the initial estimate of the support. This is a 2D boolean array.
    :param reciprocal_mask: This is the mask for the 2d detector. This algorithm only consider
                            values in the magnitude constrain with the corresponding values in this
                            mask being True.
    :param beta: The update rate in the HIO algorithm
    :param iter_num: The iteration number to apply the hio procedure
    :return:
    """
    tic = time.time()

    ###############################################################################################
    # Step 0: Generated required meta-parameters
    ###############################################################################################
    # Because later, I would need the opposite of the support, therefore I flip it here.
    flipped_support_bool = np.logical_not(support_bool)

    # Make a copy of the initial support
    initial_support = np.copy(support_bool)

    ###############################################################################################
    # Step 0: Create variables for calculation
    ###############################################################################################
    magnitude_constrain = np.ascontiguousarray(magnitude_constrain.astype(np.complex128))

    # Variable containing the diffraction satisfies the magnitude constrain
    phase_tmp = np.exp(1j * np.random.rand(*magnitude_constrain.shape) * 2 * np.pi)
    diffract_with_magnitude_constrain = np.ascontiguousarray(np.multiply(phase_tmp,
                                                                         magnitude_constrain))

    # Containing the previous result of the density function with support constrain
    density_previous = np.fft.ifftn(diffract_with_magnitude_constrain)
    density_previous = density_previous.real

    ###############################################################################################
    # Step 3: Begin calculation
    ###############################################################################################
    # Begin the loop
    for idx in range(iter_num):
        # Derive the density from the diffraction
        density_present = np.fft.ifftn(diffract_with_magnitude_constrain)

        # Get the real part
        density_present = np.real(density_present)

        # apply real space constraints
        density_present[(density_present < 0) |
                        flipped_support_bool] = (
                density_previous[(density_present < 0) |
                                 flipped_support_bool] - beta * density_present[
                    (density_present < 0) |
                    flipped_support_bool])

        # Update the previous estimation of the density for next iteration's calculation
        density_previous = np.copy(density_present)

        # Update the guess for the diffraction
        diffract_no_magnitude_constrain = np.fft.fftn(density_present)

        # apply fourier domain constraints
        phase_tmp = np.exp(1j * np.angle(diffract_no_magnitude_constrain))
        diffract_with_magnitude_constrain[reciprocal_mask] = (magnitude_constrain[reciprocal_mask] *
                                                              phase_tmp[reciprocal_mask])
    toc = time.time()
    print("It takes {:.2f} seconds to do {} iterations.".format(toc - tic, iter_num))

    ###############################################################################################
    # Step 4: Put results into a dictionary
    ###############################################################################################
    result = {'Magnitude': magnitude_constrain,
              'Initial Support': initial_support,
              'Final Support': np.logical_not(flipped_support_bool),
              'Reconstructed Density': density_present,
              'Reconstructed Diffraction Field': diffract_no_magnitude_constrain,
              'Reconstructed Magnitude Field': np.abs(diffract_no_magnitude_constrain),
              'Calculation Time (s)': toc - tic,
              'Iteration number': iter_num,
              'Sigma list': []}

    return result


def apply_hio_with_specified_initial_condition(magnitude_constrain,
                                               support_bool,
                                               reciprocal_mask,
                                               initial_diffract_field,
                                               initial_density,
                                               beta=0.8,
                                               iter_num=1000):
    """
    This function calculate the retrieved phase and the corresponding real space electron density
    in n-dimension.

    :param magnitude_constrain: This is the magnitude measured by the detector. This has to be
                                a 2d numpy array. It can be either np.float or np.complex since I
                                will convert it into complex variable at the beginning anyway.
    :param support_bool: This is the initial estimate of the support. This is a 2D boolean array.
    :param reciprocal_mask: This is the mask for the 2d detector. This algorithm only consider
                            values in the magnitude constrain with the corresponding values in this
                            mask being True.
    :param initial_diffract_field: the initial value of the diffraction field to start the
                                    searching. Notice that even though this value is not requried
                                    to be compatible with the magnitude constrain, it's recommended
                                    to be compatible.
    :param initial_density: the initial value of the density distribution. Notice that is variable
                            is used as the previous guess in this process.

    :param beta: The update rate in the HIO algorithm
    :param iter_num: The iteration number to apply the hio procedure
    :return:
    """
    tic = time.time()

    ###############################################################################################
    # Step 0: Generated required meta-parameters
    ###############################################################################################
    # Because later, I would need the opposite of the support, therefore I flip it here.
    flipped_support_bool = np.logical_not(support_bool)

    # Make a copy of the initial support
    initial_support = np.copy(support_bool)

    ###############################################################################################
    # Step 0: Create variables for calculation
    ###############################################################################################
    magnitude_constrain = np.ascontiguousarray(magnitude_constrain.astype(np.complex128))

    # Variable containing the diffraction satisfies the magnitude constrain
    phase_tmp = np.exp(1j * np.angle(initial_diffract_field))
    diffract_with_magnitude_constrain = np.ascontiguousarray(np.multiply(phase_tmp,
                                                                         magnitude_constrain))

    # Containing the previous result of the density function with support constrain
    density_previous = np.ascontiguousarray(initial_density)

    ###############################################################################################
    # Step 3: Begin calculation
    ###############################################################################################
    # Begin the loop
    for idx in range(iter_num):
        # Derive the density from the diffraction
        density_present = np.fft.ifftn(diffract_with_magnitude_constrain)

        # Get the real part
        density_present = np.real(density_present)

        # apply real space constraints
        density_present[(density_present < 0) |
                        flipped_support_bool] = (density_previous[(density_present < 0) |
                                                                  flipped_support_bool] -
                                                 beta *
                                                 density_present[
                                                     (density_present < 0) |
                                                     flipped_support_bool])

        # Update the previous estimation of the density for next iteration's calculation
        density_previous = np.copy(density_present)

        # Update the guess for the diffraction
        diffract_no_magnitude_constrain = np.fft.fftn(density_present)

        # apply fourier domain constraints
        phase_tmp = np.exp(1j * np.angle(diffract_no_magnitude_constrain))
        diffract_with_magnitude_constrain[reciprocal_mask] = (magnitude_constrain[reciprocal_mask] *
                                                              phase_tmp[reciprocal_mask])

    toc = time.time()
    print("It takes {:.2f} seconds to do {} iterations.".format(toc - tic, iter_num))

    ###############################################################################################
    # Step 4: Put results into a dictionary
    ###############################################################################################
    result = {'Magnitude': magnitude_constrain,
              'Initial Support': initial_support,
              'Final Support': np.logical_not(flipped_support_bool),
              'Reconstructed Density': density_present,
              'Reconstructed Diffraction Field': diffract_no_magnitude_constrain,
              'Reconstructed Magnitude Field': np.abs(diffract_no_magnitude_constrain),
              'Calculation Time (s)': toc - tic,
              'Iteration number': iter_num,
              'Sigma list': []}

    return result

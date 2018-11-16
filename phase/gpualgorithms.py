from numba import cuda
import numpy as np
from pyculib import fft as pfft
import scipy.ndimage as sn
import time
import math
import phase.gpuutil2d as gpuutil2d


def apply_2d_hio_with_wrap_shrink(magnitude_constrain,
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
    in the 2d case.

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
    # Step -1: Generated required meta-parameters
    ###############################################################################################
    support_decay_rate = int(support_decay_rate)

    sigma_num = int(iter_num / support_decay_rate) + 1
    sigma_list = np.linspace(sigma_start, sigma_stop, num=sigma_num, endpoint=True)

    # Make a copy of the initial support
    initial_support = np.copy(support_bool)

    ###############################################################################################
    # Step 0: Create variables for calculation
    ###############################################################################################
    shape_0, shape_1 = magnitude_constrain.shape

    magnitude_constrain = np.ascontiguousarray(magnitude_constrain.astype(np.complex128))

    # Variable containing the diffraction satisfies the magnitude constrain
    magnitude_constrain_pattern = np.asanyarray(np.zeros_like(magnitude_constrain,
                                                              dtype=np.complex128))

    # Variable holding the derived intensity from the complex diffraction field. Notice
    # That this is a complex variable
    density_no_constrain_complex = np.asanyarray(np.zeros_like(magnitude_constrain,
                                                               dtype=np.complex128))

    # Variable holding the real part of the derived density
    density_no_constrain_real = np.asanyarray(np.zeros_like(magnitude_constrain,
                                                            dtype=np.float64))

    # Variable holding the real density with support constrain
    density_with_constrain_real = np.asanyarray(np.zeros_like(magnitude_constrain,
                                                              dtype=np.float64))

    # Cast the real density with support constrain in to complex variables
    # Then this variable is used to get the updated diffraction field which is used to get
    # updated phase values.
    density_with_constrain_complex = np.asanyarray(np.zeros_like(magnitude_constrain,
                                                                 dtype=np.complex128))

    # Containing the previous result of the density function with support constrain
    density_real_previous = np.asanyarray(np.zeros_like(magnitude_constrain,
                                                        dtype=np.float64))

    # Retrieved diffraction field with phase
    diffract_field_complex = np.asanyarray(np.zeros_like(magnitude_constrain,
                                                         dtype=np.complex128))

    ###############################################################################################
    # Step 1: Configure the gpu devices
    ###############################################################################################
    # Initialize the gpu parameters
    tpb = 32

    # Configure the blocks

    # Threads per blocks
    threadspb = (tpb, tpb)

    # Blocks per grids
    blockspg_x = int(math.ceil(shape_0 / threadspb[1]))
    blockspg_y = int(math.ceil(shape_1 / threadspb[0]))
    blockspg = (blockspg_x, blockspg_y)

    ###############################################################################################
    # Step 2: Move all the variables to the gpu
    ###############################################################################################
    gpu_magnitude_constrain = cuda.to_device(magnitude_constrain)
    gpu_support_bool = cuda.to_device(support_bool)
    gpu_reciprocal_mask = cuda.to_device(reciprocal_mask)

    # Variable containing the diffraction satisfies the magnitude constrain
    gpu_magnitude_constrain_pattern = cuda.to_device(magnitude_constrain_pattern)

    # Variable holding the derived intensity from the complex diffraction field. Notice
    # That this is a complex variable
    gpu_density_no_constrain_complex = cuda.to_device(density_no_constrain_complex)

    # Variable holding the real part of the derived density
    gpu_density_no_constrain_real = cuda.to_device(density_no_constrain_real)

    # Variable holding the real density with support constrain
    gpu_density_with_constrain_real = cuda.to_device(density_with_constrain_real)

    # Cast the real density with support constrain in to complex variables
    # Then this variable is used to get the updated diffraction field which is used to get
    # updated phase values.
    gpu_density_with_constrain_complex = cuda.to_device(density_with_constrain_complex)

    # Containing the previous result of the density function with support constrain
    gpu_density_real_previous = cuda.to_device(density_real_previous)

    # Retrieved diffraction field with phase
    gpu_diffract_field_complex = cuda.to_device(diffract_field_complex)

    ###############################################################################################
    # Step 3: Begin calculation
    ###############################################################################################
    # Begin the loop
    for idx in range(iter_num):

        pfft.ifft(ary=gpu_magnitude_constrain_pattern,
                  out=gpu_density_no_constrain_complex)

        gpuutil2d.get_real_part[blockspg, threadspb](shape_0,
                                                     shape_1,
                                                     gpu_density_no_constrain_real,
                                                     gpu_density_no_constrain_complex
                                                     )

        # apply real space constraints
        gpuutil2d.apply_support_constrain[blockspg, threadspb](shape_0,
                                                               shape_1,
                                                               beta,
                                                               gpu_density_no_constrain_real,
                                                               gpu_support_bool,
                                                               gpu_density_with_constrain_real,
                                                               gpu_density_real_previous
                                                               )

        gpuutil2d.cast_to_complex[blockspg, threadspb](shape_0,
                                                       shape_1,
                                                       gpu_density_with_constrain_real,
                                                       gpu_density_with_constrain_complex
                                                       )

        # Update the guess for the diffraction
        pfft.fft(ary=gpu_density_with_constrain_complex,
                 out=gpu_diffract_field_complex)

        # apply fourier domain constraints
        gpuutil2d.apply_diffraction_constrain_mask[
            blockspg, threadspb](shape_0,
                                 shape_1,
                                 gpu_magnitude_constrain,
                                 gpu_diffract_field_complex,
                                 gpu_magnitude_constrain_pattern,
                                 gpu_reciprocal_mask)

        # Every 50 iterations, update the estimation of the support
        if np.mod(idx + 1, support_decay_rate) == 0:
            gpu_density_with_constrain_real.to_host()
            gpu_support_bool.to_host()

            filtered = sn.filters.gaussian_filter(
                input=density_with_constrain_real,
                sigma=sigma_list[int((idx + 1) / support_decay_rate)])

            # Find the largest number of pixels
            max_value = np.max(filtered)
            min_value = np.min(filtered)

            support_bool = np.zeros_like(magnitude_constrain, dtype=np.bool)
            support_bool[
                filtered > (min_value + threshold_ratio * (max_value - min_value))] = True

            gpu_support_bool = cuda.to_device(support_bool)
            gpu_density_with_constrain_real = cuda.to_device(
                density_with_constrain_real)

    # Move all the variables back to host
    gpu_magnitude_constrain.to_host()
    gpu_support_bool.to_host()
    gpu_reciprocal_mask.to_host()

    gpu_magnitude_constrain_pattern.to_host()
    gpu_density_no_constrain_complex.to_host()
    gpu_density_no_constrain_real.to_host()
    gpu_density_with_constrain_real.to_host()
    gpu_density_with_constrain_complex.to_host()
    gpu_density_real_previous.to_host()
    gpu_diffract_field_complex.to_host()

    toc = time.time()
    print("It takes {:.2f} seconds to do {} iterations.".format(toc - tic, iter_num))

    ###############################################################################################
    # Step 4: Put results into a dictionary
    ###############################################################################################
    result = {'Magnitude': magnitude_constrain,
              'Initial Support': initial_support,
              'Final Support': support_bool,
              'Reconstructed Density': density_with_constrain_real,
              'Reconstructed Diffraction Field': diffract_field_complex,
              'Reconstructed Magnitude Field': np.abs(diffract_field_complex),
              'Calculation Time (s)': toc - tic,
              'Iteration number': iter_num,
              'Sigma list': sigma_list}

    return result

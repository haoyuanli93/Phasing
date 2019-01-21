from numba import cuda
import numpy as np
from pyculib import fft as pfft
import scipy.ndimage as sn
import time
import math
import PhaseTool.gpufun2d as gpuutil2d
import PhaseTool.gpufun3d as gpuutil3d


#####################################################################################
# The following two functions are used to get the standard result
# with the standard methods and random initialization
#####################################################################################
def apply_2d_hio_with_wrap_shrink(magnitude_constrain,
                                  support_bool,
                                  reciprocal_mask,
                                  beta=0.8,
                                  iter_num=1000,
                                  threshold_ratio=0.04,
                                  sigma_start=5,
                                  sigma_stop=0.5,
                                  support_decay_rate=50,
                                  thread_per_block=32):
    """
    This function calculate the retrieved PhaseTool and the corresponding real space electron density
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
    :param thread_per_block: thread number per block. This is a gpu configuration parameter.
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
    pixel_num = float(shape_0 * shape_1)

    magnitude_constrain = np.ascontiguousarray(magnitude_constrain.astype(np.complex128))

    # Retrieved diffraction field with PhaseTool
    diffract_no_magnitude_constrain = np.ascontiguousarray(np.zeros_like(magnitude_constrain,
                                                                         dtype=np.complex128))

    # Variable containing the diffraction satisfies the magnitude constrain
    phase_tmp = np.exp(1j * np.random.rand(shape_0, shape_1) * 2 * np.pi)
    diffract_with_magnitude_constrain = np.ascontiguousarray(np.multiply(phase_tmp,
                                                                         magnitude_constrain))

    # Variable holding the derived intensity from the complex diffraction field. Notice
    # That this is a complex variable
    density_no_constrain_complex = np.ascontiguousarray(np.zeros_like(magnitude_constrain,
                                                                      dtype=np.complex128))

    # Variable holding the real part of the derived density
    density_no_constrain_real = np.ascontiguousarray(np.zeros_like(magnitude_constrain,
                                                                   dtype=np.float64))

    # Variable holding the real density with support constrain
    density_with_constrain_real = np.ascontiguousarray(np.zeros_like(magnitude_constrain,
                                                                     dtype=np.float64))

    # Cast the real density with support constrain in to complex variables
    # Then this variable is used to get the updated diffraction field which is used to get
    # updated PhaseTool values.
    density_with_constrain_complex = np.ascontiguousarray(np.zeros_like(magnitude_constrain,
                                                                        dtype=np.complex128))

    # Containing the previous result of the density function with support constrain
    density_real_previous = np.ascontiguousarray(np.zeros_like(magnitude_constrain,
                                                               dtype=np.float64))

    ###############################################################################################
    # Step 1: Configure the gpu devices
    ###############################################################################################
    # Initialize the gpu parameters
    tpb = thread_per_block

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

    # Retrieved diffraction field with PhaseTool
    gpu_diffract_no_magnitude_constrain = cuda.to_device(diffract_no_magnitude_constrain)

    # Variable containing the diffraction satisfies the magnitude constrain
    gpu_diffract_with_magnitude_constrain = cuda.to_device(diffract_with_magnitude_constrain)

    # Variable holding the derived intensity from the complex diffraction field. Notice
    # That this is a complex variable
    gpu_density_no_constrain_complex = cuda.to_device(density_no_constrain_complex)

    # Variable holding the real part of the derived density
    gpu_density_no_constrain_real = cuda.to_device(density_no_constrain_real)

    # Variable holding the real density with support constrain
    gpu_density_with_constrain_real = cuda.to_device(density_with_constrain_real)

    # Cast the real density with support constrain in to complex variables
    # Then this variable is used to get the updated diffraction field which is used to get
    # updated PhaseTool values.
    gpu_density_with_constrain_complex = cuda.to_device(density_with_constrain_complex)

    # Containing the previous result of the density function with support constrain
    gpu_density_real_previous = cuda.to_device(density_real_previous)

    ###############################################################################################
    # Step 3: Begin calculation
    ###############################################################################################
    # Begin the loop
    for idx in range(iter_num):

        pfft.ifft(ary=gpu_diffract_with_magnitude_constrain,
                  out=gpu_density_no_constrain_complex)

        gpuutil2d.get_real_part[blockspg, threadspb](shape_0,
                                                     shape_1,
                                                     pixel_num,
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
                 out=gpu_diffract_no_magnitude_constrain)

        # apply fourier domain constraints
        gpuutil2d.apply_magnitude_constrain_with_mask[
            blockspg, threadspb](shape_0,
                                 shape_1,
                                 gpu_magnitude_constrain,
                                 gpu_diffract_no_magnitude_constrain,
                                 gpu_diffract_with_magnitude_constrain,
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

    gpu_diffract_with_magnitude_constrain.to_host()
    gpu_density_no_constrain_complex.to_host()
    gpu_density_no_constrain_real.to_host()
    gpu_density_with_constrain_real.to_host()
    gpu_density_with_constrain_complex.to_host()
    gpu_density_real_previous.to_host()
    gpu_diffract_no_magnitude_constrain.to_host()

    toc = time.time()
    print("It takes {:.2f} seconds to do {} iterations.".format(toc - tic, iter_num))

    ###############################################################################################
    # Step 4: Put results into a dictionary
    ###############################################################################################
    result = {'Magnitude': magnitude_constrain,
              'Initial Support': initial_support,
              'Final Support': support_bool,
              'Reconstructed Density': density_with_constrain_real,
              'Reconstructed Diffraction Field': diffract_no_magnitude_constrain,
              'Reconstructed Magnitude Field': np.abs(diffract_no_magnitude_constrain),
              'Calculation Time (s)': toc - tic,
              'Iteration number': iter_num,
              'Sigma list': sigma_list}

    return result


def apply_3d_hio_with_wrap_shrink(magnitude_constrain,
                                  support_bool,
                                  reciprocal_mask,
                                  beta=0.8,
                                  iter_num=1000,
                                  threshold_ratio=0.04,
                                  sigma_start=5,
                                  sigma_stop=0.5,
                                  support_decay_rate=50,
                                  thread_per_block=4):
    """
    This function calculate the retrieved PhaseTool and the corresponding real space electron density
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
    :param thread_per_block: thread number per block. This is a gpu calculation parameter.
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
    shape_0, shape_1, shape_2 = magnitude_constrain.shape
    pixel_num = float(shape_0 * shape_1 * shape_2)

    magnitude_constrain = np.ascontiguousarray(magnitude_constrain.astype(np.complex128))

    # Retrieved diffraction field with PhaseTool
    diffract_no_magnitude_constrain = np.ascontiguousarray(np.zeros_like(magnitude_constrain,
                                                                         dtype=np.complex128))

    # Variable containing the diffraction satisfies the magnitude constrain
    phase_tmp = np.exp(1j * np.random.rand(shape_0, shape_1, shape_2) * 2 * np.pi)
    diffract_with_magnitude_constrain = np.ascontiguousarray(np.multiply(phase_tmp,
                                                                         magnitude_constrain))

    # Variable holding the derived intensity from the complex diffraction field. Notice
    # That this is a complex variable
    density_no_constrain_complex = np.ascontiguousarray(np.zeros_like(magnitude_constrain,
                                                                      dtype=np.complex128))

    # Variable holding the real part of the derived density
    density_no_constrain_real = np.ascontiguousarray(np.zeros_like(magnitude_constrain,
                                                                   dtype=np.float64))

    # Variable holding the real density with support constrain
    density_with_constrain_real = np.ascontiguousarray(np.zeros_like(magnitude_constrain,
                                                                     dtype=np.float64))

    # Cast the real density with support constrain in to complex variables
    # Then this variable is used to get the updated diffraction field which is used to get
    # updated PhaseTool values.
    density_with_constrain_complex = np.ascontiguousarray(np.zeros_like(magnitude_constrain,
                                                                        dtype=np.complex128))

    # Containing the previous result of the density function with support constrain
    density_real_previous = np.ascontiguousarray(np.zeros_like(magnitude_constrain,
                                                               dtype=np.float64))

    ###############################################################################################
    # Step 1: Configure the gpu devices
    ###############################################################################################
    # Initialize the gpu parameters
    tpb = thread_per_block

    # Configure the blocks

    # Threads per blocks
    threadspb = (tpb, tpb, tpb)

    # Blocks per grids
    blockspg_x = int(math.ceil(shape_0 / threadspb[0]))
    blockspg_y = int(math.ceil(shape_1 / threadspb[1]))
    blockspg_z = int(math.ceil(shape_2 / threadspb[2]))
    blockspg = (blockspg_x, blockspg_y, blockspg_z)

    ###############################################################################################
    # Step 2: Move all the variables to the gpu
    ###############################################################################################
    gpu_magnitude_constrain = cuda.to_device(magnitude_constrain)
    gpu_support_bool = cuda.to_device(support_bool)
    gpu_reciprocal_mask = cuda.to_device(reciprocal_mask)

    # Retrieved diffraction field with PhaseTool
    gpu_diffract_no_magnitude_constrain = cuda.to_device(diffract_no_magnitude_constrain)

    # Variable containing the diffraction satisfies the magnitude constrain
    gpu_diffract_with_magnitude_constrain = cuda.to_device(diffract_with_magnitude_constrain)

    # Variable holding the derived intensity from the complex diffraction field. Notice
    # That this is a complex variable
    gpu_density_no_constrain_complex = cuda.to_device(density_no_constrain_complex)

    # Variable holding the real part of the derived density
    gpu_density_no_constrain_real = cuda.to_device(density_no_constrain_real)

    # Variable holding the real density with support constrain
    gpu_density_with_constrain_real = cuda.to_device(density_with_constrain_real)

    # Cast the real density with support constrain in to complex variables
    # Then this variable is used to get the updated diffraction field which is used to get
    # updated PhaseTool values.
    gpu_density_with_constrain_complex = cuda.to_device(density_with_constrain_complex)

    # Containing the previous result of the density function with support constrain
    gpu_density_real_previous = cuda.to_device(density_real_previous)

    ###############################################################################################
    # Step 3: Begin calculation
    ###############################################################################################
    # Begin the loop
    for idx in range(iter_num):

        pfft.ifft(ary=gpu_diffract_with_magnitude_constrain,
                  out=gpu_density_no_constrain_complex)

        gpuutil3d.get_real_part[blockspg, threadspb](shape_0,
                                                     shape_1,
                                                     shape_2,
                                                     pixel_num,
                                                     gpu_density_no_constrain_real,
                                                     gpu_density_no_constrain_complex
                                                     )

        # apply real space constraints
        gpuutil3d.apply_support_constrain[blockspg, threadspb](shape_0,
                                                               shape_1,
                                                               shape_2,
                                                               beta,
                                                               gpu_density_no_constrain_real,
                                                               gpu_support_bool,
                                                               gpu_density_with_constrain_real,
                                                               gpu_density_real_previous
                                                               )

        gpuutil3d.cast_to_complex[blockspg, threadspb](shape_0,
                                                       shape_1,
                                                       shape_2,
                                                       gpu_density_with_constrain_real,
                                                       gpu_density_with_constrain_complex
                                                       )

        # Update the guess for the diffraction
        pfft.fft(ary=gpu_density_with_constrain_complex,
                 out=gpu_diffract_no_magnitude_constrain)

        # apply fourier domain constraints
        gpuutil3d.apply_magnitude_constrain_with_mask[blockspg, threadspb](
            shape_0,
            shape_1,
            shape_2,
            gpu_magnitude_constrain,
            gpu_diffract_no_magnitude_constrain,
            gpu_diffract_with_magnitude_constrain,
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

    gpu_diffract_with_magnitude_constrain.to_host()
    gpu_density_no_constrain_complex.to_host()
    gpu_density_no_constrain_real.to_host()
    gpu_density_with_constrain_real.to_host()
    gpu_density_with_constrain_complex.to_host()
    gpu_density_real_previous.to_host()
    gpu_diffract_no_magnitude_constrain.to_host()

    toc = time.time()
    print("It takes {:.2f} seconds to do {} iterations.".format(toc - tic, iter_num))

    ###############################################################################################
    # Step 4: Put results into a dictionary
    ###############################################################################################
    result = {'Magnitude': magnitude_constrain,
              'Initial Support': initial_support,
              'Final Support': support_bool,
              'Reconstructed Density': density_with_constrain_real,
              'Reconstructed Diffraction Field': diffract_no_magnitude_constrain,
              'Reconstructed Magnitude Field': np.abs(diffract_no_magnitude_constrain),
              'Calculation Time (s)': toc - tic,
              'Iteration number': iter_num,
              'Sigma list': sigma_list}

    return result


#############################################################################
# Sometimes, the researchers are very confident about the support
# In this case, there is no need to use the shrinkwrap method
#############################################################################
def apply_2d_hio_no_wrap_shrink(magnitude_constrain,
                                support_bool,
                                reciprocal_mask,
                                beta=0.8,
                                iter_num=100,
                                thread_per_block=32):
    """
    This function calculate the retrieved PhaseTool and the corresponding real space electron density
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
    :param thread_per_block: thread number per block. This is a gpu calculation parameter.
    :return:
    """
    tic = time.time()

    ###############################################################################################
    # Step -1: Generated required meta-parameters
    ###############################################################################################
    # Make a copy of the initial support
    initial_support = np.copy(support_bool)

    ###############################################################################################
    # Step 0: Create variables for calculation
    ###############################################################################################
    shape_0, shape_1 = magnitude_constrain.shape
    pixel_num = shape_0 * shape_1

    magnitude_constrain = np.ascontiguousarray(magnitude_constrain.astype(np.complex128))

    # Retrieved diffraction field with PhaseTool
    diffract_no_magnitude_constrain = np.ascontiguousarray(np.zeros_like(magnitude_constrain,
                                                                         dtype=np.complex128))

    # Variable containing the diffraction satisfies the magnitude constrain
    phase_tmp = np.exp(1j * np.random.rand(shape_0, shape_1) * 2 * np.pi)
    diffract_with_magnitude_constrain = np.ascontiguousarray(np.multiply(phase_tmp,
                                                                         magnitude_constrain))

    # Variable holding the derived intensity from the complex diffraction field. Notice
    # That this is a complex variable
    density_no_constrain_complex = np.ascontiguousarray(np.zeros_like(magnitude_constrain,
                                                                      dtype=np.complex128))

    # Variable holding the real part of the derived density
    density_no_constrain_real = np.ascontiguousarray(np.zeros_like(magnitude_constrain,
                                                                   dtype=np.float64))

    # Variable holding the real density with support constrain
    density_with_constrain_real = np.ascontiguousarray(np.zeros_like(magnitude_constrain,
                                                                     dtype=np.float64))

    # Cast the real density with support constrain in to complex variables
    # Then this variable is used to get the updated diffraction field which is used to get
    # updated PhaseTool values.
    density_with_constrain_complex = np.ascontiguousarray(np.zeros_like(magnitude_constrain,
                                                                        dtype=np.complex128))

    # Containing the previous result of the density function with support constrain
    density_real_previous = np.ascontiguousarray(np.zeros_like(magnitude_constrain,
                                                               dtype=np.float64))

    ###############################################################################################
    # Step 1: Configure the gpu devices
    ###############################################################################################
    # Initialize the gpu parameters
    tpb = thread_per_block

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

    # Retrieved diffraction field with PhaseTool
    gpu_diffract_no_magnitude_constrain = cuda.to_device(diffract_no_magnitude_constrain)

    # Variable containing the diffraction satisfies the magnitude constrain
    gpu_diffract_with_magnitude_constrain = cuda.to_device(diffract_with_magnitude_constrain)

    # Variable holding the derived intensity from the complex diffraction field. Notice
    # That this is a complex variable
    gpu_density_no_constrain_complex = cuda.to_device(density_no_constrain_complex)

    # Variable holding the real part of the derived density
    gpu_density_no_constrain_real = cuda.to_device(density_no_constrain_real)

    # Variable holding the real density with support constrain
    gpu_density_with_constrain_real = cuda.to_device(density_with_constrain_real)

    # Cast the real density with support constrain in to complex variables
    # Then this variable is used to get the updated diffraction field which is used to get
    # updated PhaseTool values.
    gpu_density_with_constrain_complex = cuda.to_device(density_with_constrain_complex)

    # Containing the previous result of the density function with support constrain
    gpu_density_real_previous = cuda.to_device(density_real_previous)

    ###############################################################################################
    # Step 3: Begin calculation
    ###############################################################################################
    # Begin the loop
    for idx in range(iter_num):
        pfft.ifft(ary=gpu_diffract_with_magnitude_constrain,
                  out=gpu_density_no_constrain_complex)

        gpuutil2d.get_real_part[blockspg, threadspb](shape_0,
                                                     shape_1,
                                                     pixel_num,
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
                 out=gpu_diffract_no_magnitude_constrain)

        # apply fourier domain constraints
        gpuutil2d.apply_magnitude_constrain_with_mask[
            blockspg, threadspb](shape_0,
                                 shape_1,
                                 gpu_magnitude_constrain,
                                 gpu_diffract_no_magnitude_constrain,
                                 gpu_diffract_with_magnitude_constrain,
                                 gpu_reciprocal_mask)

    # Move all the variables back to host
    gpu_magnitude_constrain.to_host()
    gpu_support_bool.to_host()
    gpu_reciprocal_mask.to_host()

    gpu_diffract_with_magnitude_constrain.to_host()
    gpu_density_no_constrain_complex.to_host()
    gpu_density_no_constrain_real.to_host()
    gpu_density_with_constrain_real.to_host()
    gpu_density_with_constrain_complex.to_host()
    gpu_density_real_previous.to_host()
    gpu_diffract_no_magnitude_constrain.to_host()

    toc = time.time()
    print("It takes {:.2f} seconds to do {} iterations.".format(toc - tic, iter_num))

    ###############################################################################################
    # Step 4: Put results into a dictionary
    ###############################################################################################
    result = {'Magnitude': magnitude_constrain,
              'Initial Support': initial_support,
              'Final Support': support_bool,
              'Reconstructed Density': density_with_constrain_real,
              'Reconstructed Diffraction Field': diffract_no_magnitude_constrain,
              'Reconstructed Magnitude Field': np.abs(diffract_no_magnitude_constrain),
              'Calculation Time (s)': toc - tic,
              'Iteration number': iter_num,
              'Sigma list': []
              }

    return result


def apply_3d_hio_no_wrap_shrink(magnitude_constrain,
                                support_bool,
                                reciprocal_mask,
                                beta=0.8,
                                iter_num=100,
                                thread_per_block=4):
    """
    This function calculate the retrieved PhaseTool and the corresponding real space electron density
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
    :param thread_per_block: thread number per block. This is a gpu calculation parameter.

    :return:
    """
    tic = time.time()

    ###############################################################################################
    # Step -1: Generated required meta-parameters
    ###############################################################################################
    # Make a copy of the initial support
    initial_support = np.copy(support_bool)

    ###############################################################################################
    # Step 0: Create variables for calculation
    ###############################################################################################
    shape_0, shape_1, shape_2 = magnitude_constrain.shape
    pixel_num = shape_0 * shape_1 * shape_2

    magnitude_constrain = np.ascontiguousarray(magnitude_constrain.astype(np.complex128))

    # Retrieved diffraction field with PhaseTool
    diffract_no_magnitude_constrain = np.ascontiguousarray(np.zeros_like(magnitude_constrain,
                                                                         dtype=np.complex128))

    # Variable containing the diffraction satisfies the magnitude constrain
    phase_tmp = np.exp(1j * np.random.rand(shape_0, shape_1, shape_2) * 2 * np.pi)
    diffract_with_magnitude_constrain = np.ascontiguousarray(np.multiply(phase_tmp,
                                                                         magnitude_constrain))

    # Variable holding the derived intensity from the complex diffraction field. Notice
    # That this is a complex variable
    density_no_constrain_complex = np.ascontiguousarray(np.zeros_like(magnitude_constrain,
                                                                      dtype=np.complex128))

    # Variable holding the real part of the derived density
    density_no_constrain_real = np.ascontiguousarray(np.zeros_like(magnitude_constrain,
                                                                   dtype=np.float64))

    # Variable holding the real density with support constrain
    density_with_constrain_real = np.ascontiguousarray(np.zeros_like(magnitude_constrain,
                                                                     dtype=np.float64))

    # Cast the real density with support constrain in to complex variables
    # Then this variable is used to get the updated diffraction field which is used to get
    # updated PhaseTool values.
    density_with_constrain_complex = np.ascontiguousarray(np.zeros_like(magnitude_constrain,
                                                                        dtype=np.complex128))

    # Containing the previous result of the density function with support constrain
    density_real_previous = np.ascontiguousarray(np.zeros_like(magnitude_constrain,
                                                               dtype=np.float64))

    ###############################################################################################
    # Step 1: Configure the gpu devices
    ###############################################################################################
    # Initialize the gpu parameters
    tpb = thread_per_block

    # Configure the blocks

    # Threads per blocks
    threadspb = (tpb, tpb, tpb)

    # Blocks per grids
    blockspg_x = int(math.ceil(shape_0 / threadspb[0]))
    blockspg_y = int(math.ceil(shape_1 / threadspb[1]))
    blockspg_z = int(math.ceil(shape_2 / threadspb[2]))
    blockspg = (blockspg_x, blockspg_y, blockspg_z)

    ###############################################################################################
    # Step 2: Move all the variables to the gpu
    ###############################################################################################
    gpu_magnitude_constrain = cuda.to_device(magnitude_constrain)
    gpu_support_bool = cuda.to_device(support_bool)
    gpu_reciprocal_mask = cuda.to_device(reciprocal_mask)

    # Retrieved diffraction field with PhaseTool
    gpu_diffract_no_magnitude_constrain = cuda.to_device(diffract_no_magnitude_constrain)

    # Variable containing the diffraction satisfies the magnitude constrain
    gpu_diffract_with_magnitude_constrain = cuda.to_device(diffract_with_magnitude_constrain)

    # Variable holding the derived intensity from the complex diffraction field. Notice
    # That this is a complex variable
    gpu_density_no_constrain_complex = cuda.to_device(density_no_constrain_complex)

    # Variable holding the real part of the derived density
    gpu_density_no_constrain_real = cuda.to_device(density_no_constrain_real)

    # Variable holding the real density with support constrain
    gpu_density_with_constrain_real = cuda.to_device(density_with_constrain_real)

    # Cast the real density with support constrain in to complex variables
    # Then this variable is used to get the updated diffraction field which is used to get
    # updated PhaseTool values.
    gpu_density_with_constrain_complex = cuda.to_device(density_with_constrain_complex)

    # Containing the previous result of the density function with support constrain
    gpu_density_real_previous = cuda.to_device(density_real_previous)

    ###############################################################################################
    # Step 3: Begin calculation
    ###############################################################################################
    # Begin the loop
    for idx in range(iter_num):
        pfft.ifft(ary=gpu_diffract_with_magnitude_constrain,
                  out=gpu_density_no_constrain_complex)

        gpuutil3d.get_real_part[blockspg, threadspb](shape_0,
                                                     shape_1,
                                                     shape_2,
                                                     pixel_num,
                                                     gpu_density_no_constrain_real,
                                                     gpu_density_no_constrain_complex
                                                     )

        # apply real space constraints
        gpuutil3d.apply_support_constrain[blockspg, threadspb](shape_0,
                                                               shape_1,
                                                               shape_2,
                                                               beta,
                                                               gpu_density_no_constrain_real,
                                                               gpu_support_bool,
                                                               gpu_density_with_constrain_real,
                                                               gpu_density_real_previous
                                                               )

        gpuutil3d.cast_to_complex[blockspg, threadspb](shape_0,
                                                       shape_1,
                                                       shape_2,
                                                       gpu_density_with_constrain_real,
                                                       gpu_density_with_constrain_complex
                                                       )

        # Update the guess for the diffraction
        pfft.fft(ary=gpu_density_with_constrain_complex,
                 out=gpu_diffract_no_magnitude_constrain)

        # apply fourier domain constraints
        gpuutil3d.apply_magnitude_constrain_with_mask[blockspg, threadspb](
            shape_0,
            shape_1,
            shape_2,
            gpu_magnitude_constrain,
            gpu_diffract_no_magnitude_constrain,
            gpu_diffract_with_magnitude_constrain,
            gpu_reciprocal_mask)

    # Move all the variables back to host
    gpu_magnitude_constrain.to_host()
    gpu_support_bool.to_host()
    gpu_reciprocal_mask.to_host()

    gpu_diffract_with_magnitude_constrain.to_host()
    gpu_density_no_constrain_complex.to_host()
    gpu_density_no_constrain_real.to_host()
    gpu_density_with_constrain_real.to_host()
    gpu_density_with_constrain_complex.to_host()
    gpu_density_real_previous.to_host()
    gpu_diffract_no_magnitude_constrain.to_host()

    toc = time.time()
    print("It takes {:.2f} seconds to do {} iterations.".format(toc - tic, iter_num))

    ###############################################################################################
    # Step 4: Put results into a dictionary
    ###############################################################################################
    result = {'Magnitude': magnitude_constrain,
              'Initial Support': initial_support,
              'Final Support': support_bool,
              'Reconstructed Density': density_with_constrain_real,
              'Reconstructed Diffraction Field': diffract_no_magnitude_constrain,
              'Reconstructed Magnitude Field': np.abs(diffract_no_magnitude_constrain),
              'Calculation Time (s)': toc - tic,
              'Iteration number': iter_num,
              'Sigma list': []}

    return result


############################################################################
# Some times, one might want to tune the support very carefully. Or one
# might want to try some patching on the magnitude pattern to fix some
# missing region. The functions in below only do the HIO with the specified
# initial density and diffraction field which is suitable for the purpose.
############################################################################
def apply_2d_hio_with_specified_initial_condition(magnitude_constrain,
                                                  support_bool,
                                                  reciprocal_mask,
                                                  initial_diffract_field,
                                                  initial_density,
                                                  beta=0.8,
                                                  iter_num=100,
                                                  thread_per_block=32):
    """
    This function calculate the retrieved PhaseTool and the corresponding real space electron density
    in the 2d case.

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
    :param thread_per_block: thread number per block. This is a gpu calculation parameter.
    :return:
    """
    tic = time.time()

    ###############################################################################################
    # Step -1: Generated required meta-parameters
    ###############################################################################################
    # Make a copy of the initial support
    initial_support = np.copy(support_bool)

    ###############################################################################################
    # Step 0: Create variables for calculation
    ###############################################################################################
    shape_0, shape_1 = magnitude_constrain.shape
    pixel_num = shape_0 * shape_1

    magnitude_constrain = np.ascontiguousarray(magnitude_constrain.astype(np.complex128))

    # Retrieved diffraction field with PhaseTool
    diffract_no_magnitude_constrain = np.ascontiguousarray(
        initial_diffract_field.astype(np.complex128))

    # Variable containing the diffraction satisfies the magnitude constrain
    phase_tmp = np.exp(1j * np.angle(diffract_no_magnitude_constrain))
    diffract_with_magnitude_constrain = np.ascontiguousarray(np.multiply(phase_tmp,
                                                                         magnitude_constrain))

    # Variable holding the derived intensity from the complex diffraction field. Notice
    # That this is a complex variable
    density_no_constrain_complex = np.ascontiguousarray(np.zeros_like(magnitude_constrain,
                                                                      dtype=np.complex128))

    # Variable holding the real part of the derived density
    density_no_constrain_real = np.ascontiguousarray(np.zeros_like(magnitude_constrain,
                                                                   dtype=np.float64))

    # Variable holding the real density with support constrain
    density_with_constrain_real = np.ascontiguousarray(np.zeros_like(magnitude_constrain,
                                                                     dtype=np.float64))

    # Cast the real density with support constrain in to complex variables
    # Then this variable is used to get the updated diffraction field which is used to get
    # updated PhaseTool values.
    density_with_constrain_complex = np.ascontiguousarray(np.zeros_like(magnitude_constrain,
                                                                        dtype=np.complex128))

    # Containing the previous result of the density function with support constrain
    density_real_previous = np.ascontiguousarray(np.copy(initial_density))

    ###############################################################################################
    # Step 1: Configure the gpu devices
    ###############################################################################################
    # Initialize the gpu parameters
    tpb = thread_per_block

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

    # Retrieved diffraction field with PhaseTool
    gpu_diffract_no_magnitude_constrain = cuda.to_device(diffract_no_magnitude_constrain)

    # Variable containing the diffraction satisfies the magnitude constrain
    gpu_diffract_with_magnitude_constrain = cuda.to_device(diffract_with_magnitude_constrain)

    # Variable holding the derived intensity from the complex diffraction field. Notice
    # That this is a complex variable
    gpu_density_no_constrain_complex = cuda.to_device(density_no_constrain_complex)

    # Variable holding the real part of the derived density
    gpu_density_no_constrain_real = cuda.to_device(density_no_constrain_real)

    # Variable holding the real density with support constrain
    gpu_density_with_constrain_real = cuda.to_device(density_with_constrain_real)

    # Cast the real density with support constrain in to complex variables
    # Then this variable is used to get the updated diffraction field which is used to get
    # updated PhaseTool values.
    gpu_density_with_constrain_complex = cuda.to_device(density_with_constrain_complex)

    # Containing the previous result of the density function with support constrain
    gpu_density_real_previous = cuda.to_device(density_real_previous)

    ###############################################################################################
    # Step 3: Begin calculation
    ###############################################################################################
    # Begin the loop
    for idx in range(iter_num):
        pfft.ifft(ary=gpu_diffract_with_magnitude_constrain,
                  out=gpu_density_no_constrain_complex)

        gpuutil2d.get_real_part[blockspg, threadspb](shape_0,
                                                     shape_1,
                                                     pixel_num,
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
                 out=gpu_diffract_no_magnitude_constrain)

        # apply fourier domain constraints
        gpuutil2d.apply_magnitude_constrain_with_mask[
            blockspg, threadspb](shape_0,
                                 shape_1,
                                 gpu_magnitude_constrain,
                                 gpu_diffract_no_magnitude_constrain,
                                 gpu_diffract_with_magnitude_constrain,
                                 gpu_reciprocal_mask)

    # Move all the variables back to host
    gpu_magnitude_constrain.to_host()
    gpu_support_bool.to_host()
    gpu_reciprocal_mask.to_host()

    gpu_diffract_with_magnitude_constrain.to_host()
    gpu_density_no_constrain_complex.to_host()
    gpu_density_no_constrain_real.to_host()
    gpu_density_with_constrain_real.to_host()
    gpu_density_with_constrain_complex.to_host()
    gpu_density_real_previous.to_host()
    gpu_diffract_no_magnitude_constrain.to_host()

    toc = time.time()
    print("It takes {:.2f} seconds to do {} iterations.".format(toc - tic, iter_num))

    ###############################################################################################
    # Step 4: Put results into a dictionary
    ###############################################################################################
    result = {'Magnitude': magnitude_constrain,
              'Initial Support': initial_support,
              'Final Support': support_bool,
              'Reconstructed Density': density_with_constrain_real,
              'Reconstructed Diffraction Field': diffract_no_magnitude_constrain,
              'Reconstructed Magnitude Field': np.abs(diffract_no_magnitude_constrain),
              'Calculation Time (s)': toc - tic,
              'Iteration number': iter_num,
              'Sigma list': []
              }

    return result


def apply_3d_hio_with_specified_initial_condition(magnitude_constrain,
                                                  support_bool,
                                                  reciprocal_mask,
                                                  initial_diffract_field,
                                                  initial_density,
                                                  beta=0.8,
                                                  iter_num=100,
                                                  thread_per_block=4):
    """
    This function calculate the retrieved PhaseTool and the corresponding real space electron density
    in the 2d case.

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
    :param thread_per_block: thread number per block. This is a gpu calculation parameter.

    :return:
    """
    tic = time.time()

    ###############################################################################################
    # Step -1: Generated required meta-parameters
    ###############################################################################################
    # Make a copy of the initial support
    initial_support = np.copy(support_bool)

    ###############################################################################################
    # Step 0: Create variables for calculation
    ###############################################################################################
    shape_0, shape_1, shape_2 = magnitude_constrain.shape
    pixel_num = shape_0 * shape_1 * shape_2

    magnitude_constrain = np.ascontiguousarray(magnitude_constrain.astype(np.complex128))

    # Retrieved diffraction field with PhaseTool
    diffract_no_magnitude_constrain = np.ascontiguousarray(
        initial_diffract_field.astype(np.complex128))

    # Variable containing the diffraction satisfies the magnitude constrain
    phase_tmp = np.exp(1j * np.angle(diffract_no_magnitude_constrain))
    diffract_with_magnitude_constrain = np.ascontiguousarray(np.multiply(phase_tmp,
                                                                         magnitude_constrain))

    # Variable holding the derived intensity from the complex diffraction field. Notice
    # That this is a complex variable
    density_no_constrain_complex = np.ascontiguousarray(np.zeros_like(magnitude_constrain,
                                                                      dtype=np.complex128))

    # Variable holding the real part of the derived density
    density_no_constrain_real = np.ascontiguousarray(np.zeros_like(magnitude_constrain,
                                                                   dtype=np.float64))

    # Variable holding the real density with support constrain
    density_with_constrain_real = np.ascontiguousarray(np.zeros_like(magnitude_constrain,
                                                                     dtype=np.float64))

    # Cast the real density with support constrain in to complex variables
    # Then this variable is used to get the updated diffraction field which is used to get
    # updated PhaseTool values.
    density_with_constrain_complex = np.ascontiguousarray(np.zeros_like(magnitude_constrain,
                                                                        dtype=np.complex128))

    # Containing the previous result of the density function with support constrain
    density_real_previous = np.ascontiguousarray(np.copy(initial_density))

    ###############################################################################################
    # Step 1: Configure the gpu devices
    ###############################################################################################
    # Initialize the gpu parameters
    tpb = thread_per_block

    # Configure the blocks

    # Threads per blocks
    threadspb = (tpb, tpb, tpb)

    # Blocks per grids
    blockspg_x = int(math.ceil(shape_0 / threadspb[0]))
    blockspg_y = int(math.ceil(shape_1 / threadspb[1]))
    blockspg_z = int(math.ceil(shape_2 / threadspb[2]))
    blockspg = (blockspg_x, blockspg_y, blockspg_z)

    ###############################################################################################
    # Step 2: Move all the variables to the gpu
    ###############################################################################################
    gpu_magnitude_constrain = cuda.to_device(magnitude_constrain)
    gpu_support_bool = cuda.to_device(support_bool)
    gpu_reciprocal_mask = cuda.to_device(reciprocal_mask)

    # Retrieved diffraction field with PhaseTool
    gpu_diffract_no_magnitude_constrain = cuda.to_device(diffract_no_magnitude_constrain)

    # Variable containing the diffraction satisfies the magnitude constrain
    gpu_diffract_with_magnitude_constrain = cuda.to_device(diffract_with_magnitude_constrain)

    # Variable holding the derived intensity from the complex diffraction field. Notice
    # That this is a complex variable
    gpu_density_no_constrain_complex = cuda.to_device(density_no_constrain_complex)

    # Variable holding the real part of the derived density
    gpu_density_no_constrain_real = cuda.to_device(density_no_constrain_real)

    # Variable holding the real density with support constrain
    gpu_density_with_constrain_real = cuda.to_device(density_with_constrain_real)

    # Cast the real density with support constrain in to complex variables
    # Then this variable is used to get the updated diffraction field which is used to get
    # updated PhaseTool values.
    gpu_density_with_constrain_complex = cuda.to_device(density_with_constrain_complex)

    # Containing the previous result of the density function with support constrain
    gpu_density_real_previous = cuda.to_device(density_real_previous)

    ###############################################################################################
    # Step 3: Begin calculation
    ###############################################################################################
    # Begin the loop
    for idx in range(iter_num):
        pfft.ifft(ary=gpu_diffract_with_magnitude_constrain,
                  out=gpu_density_no_constrain_complex)

        gpuutil3d.get_real_part[blockspg, threadspb](shape_0,
                                                     shape_1,
                                                     shape_2,
                                                     pixel_num,
                                                     gpu_density_no_constrain_real,
                                                     gpu_density_no_constrain_complex
                                                     )

        # apply real space constraints
        gpuutil3d.apply_support_constrain[blockspg, threadspb](shape_0,
                                                               shape_1,
                                                               shape_2,
                                                               beta,
                                                               gpu_density_no_constrain_real,
                                                               gpu_support_bool,
                                                               gpu_density_with_constrain_real,
                                                               gpu_density_real_previous
                                                               )

        gpuutil3d.cast_to_complex[blockspg, threadspb](shape_0,
                                                       shape_1,
                                                       shape_2,
                                                       gpu_density_with_constrain_real,
                                                       gpu_density_with_constrain_complex
                                                       )

        # Update the guess for the diffraction
        pfft.fft(ary=gpu_density_with_constrain_complex,
                 out=gpu_diffract_no_magnitude_constrain)

        # apply fourier domain constraints
        gpuutil3d.apply_magnitude_constrain_with_mask[blockspg, threadspb](
            shape_0,
            shape_1,
            shape_2,
            gpu_magnitude_constrain,
            gpu_diffract_no_magnitude_constrain,
            gpu_diffract_with_magnitude_constrain,
            gpu_reciprocal_mask)

    # Move all the variables back to host
    gpu_magnitude_constrain.to_host()
    gpu_support_bool.to_host()
    gpu_reciprocal_mask.to_host()

    gpu_diffract_with_magnitude_constrain.to_host()
    gpu_density_no_constrain_complex.to_host()
    gpu_density_no_constrain_real.to_host()
    gpu_density_with_constrain_real.to_host()
    gpu_density_with_constrain_complex.to_host()
    gpu_density_real_previous.to_host()
    gpu_diffract_no_magnitude_constrain.to_host()

    toc = time.time()
    print("It takes {:.2f} seconds to do {} iterations.".format(toc - tic, iter_num))

    ###############################################################################################
    # Step 4: Put results into a dictionary
    ###############################################################################################
    result = {'Magnitude': magnitude_constrain,
              'Initial Support': initial_support,
              'Final Support': support_bool,
              'Reconstructed Density': density_with_constrain_real,
              'Reconstructed Diffraction Field': diffract_no_magnitude_constrain,
              'Reconstructed Magnitude Field': np.abs(diffract_no_magnitude_constrain),
              'Calculation Time (s)': toc - tic,
              'Iteration number': iter_num,
              'Sigma list': []}

    return result

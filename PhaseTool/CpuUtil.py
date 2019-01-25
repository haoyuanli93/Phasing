import numpy as np


def iterative_projection_normal(data_dict, holder_dict, a, b, c, d, e, f):
    """
    This function carries out the iterative projection for one iteration.

    :param data_dict: This dictionary contains the following info

                        magnitude array      -> The numpy array containing the magnitude array

                        magnitude mask       -> The boolean mask for the magnitude array

                        support              -> The boolean array for the support

                        diffraction      -> The diffraction field from previous step
                                            Notice that this is not essential. I include it
                                            here only becase it can be useful for later usage.

                        density          -> The density from previous iteration


    :param holder_dict:
                        This dictionary contains intermediate results to reduce memory allocation
                        time.

                        new diffraction with magnitude constrain  -> This is the diffraction field
                                                                     with magnitude constrain

                        new diffraction magnitude   -> This is the magnitude of the diffraction
                                                        field before applying the magnitude
                                                        constrain

                        new density tmp      -> This is the new density derived from the diffraction
                                                field with the magnitude constrain

                        phase holder          -> This is the PhaseTool of the derived diffraction
                                                    field

                        support not           -> not*support

                        magnitude mask not   -> The boolean mask for the magnitude array.
                                                This is simply
                                                        not*magnitude_mask

                        modified support   -> In the algorithm, one needs to change pixels
                                                     in the support and in the same time
                                                     satisfied some conditions

                        modified support not  ->  np.logical_not(modified support)

                        support holder temporary  ->  A temporary holder when calculating the
                                                     modified support.

                        tmp holder 2        -> Sorry I just really can not think up a name for
                                                this holder. This is for the approximated
                                                magnitude projection operator.
                        tmp holder 3        -> Sorry I just really can not think up a name for
                                                this holder. This is for the approximated
                                                magnitude projection operator.
                        tmp holder 4        -> Sorry I just really can not think up a name for
                                                this holder. This is for the approximated
                                                magnitude projection operator.
    :param a:
    :param b:
    :param c:
    :param d:
    :param e:
    :param f:
    
                        These six parameters are complicated.

                        epsilon : This is the parameter for the approximation


                        par_a
                        par_b
                        par_c
                        par_d
                        par_e
                        par_f


                       The last six entries are complicated. They defines the structure of the
                       projections.

                       Below, I use u_(n+1) to denote the new density, then in general,
                       the algorithm can be represented as

                                     par_c * P(u_n)(x) + par_d * u_n(x)
                       u_(n+1)(x) =
                                     par_a * P(u_n)(x) + par_b * u_n(x)     for x (in support) and
                                                                            par_e * P(u_n)(x) >
                                                                            par_f * u_n(x)

    :return: None. The resut is directly saved to the dictionary. This is also the very reason
            why I choose this mutable structure.
    """

    # Because this is a very basic function, I will not check paramters or counting the time

    # Get data variables
    mag = data_dict['magnitude array']
    mag_m = data_dict['magnitude mask']

    density = data_dict['density']
    support = data_dict['support']

    # Get holder variables
    support_m = holder_dict['modified support']
    support_mn = holder_dict['modified support not']
    support_t = holder_dict['support holder temporary']

    ndens_t = holder_dict['new density tmp']

    ndiff_m = holder_dict['new diffraction magnitude']
    ndiff_c = holder_dict['new diffraction with magnitude constrain']
    phase = holder_dict['phase holder']

    # Step 1: Calculate the fourier transformation of the density
    ndiff_c[:] = np.fft.fftn(density)

    # Step 2: Apply magnitude constrain to the diffraction
    np.absolute(np.absolute(ndiff_c[mag_m]), out=ndiff_m[mag_m])
    np.divide(ndiff_c[mag_m], ndiff_m[mag_m],
              out=phase[mag_m], where=ndiff_m[mag_m] > 0)

    np.multiply(mag[mag_m], phase[mag_m], out=ndiff_c[mag_m])

    # Step 3: Get the updated density
    ndens_t[:] = np.fft.ifftn(ndiff_c).real

    # Step 4: Apply real space constrain
    # Get the positions where to modify
    support_m[:] = support[:]

    np.add(e * ndens_t[support], f * density[support], out=support_t[support])
    np.greater(support_t[support], 0, out=support_m[support])

    # Update the modified support not
    np.logical_not(support_m, out=support_mn)

    # Apply the real space update rule
    np.add(c * ndens_t[support_mn], d * density[support_mn], out=density[support_mn])
    np.add(a * ndens_t[support_m], b * density[support_m], out=density[support_m])


def error_reduction(data_dict, holder_dict):
    """
    This function carries out the iterative projection for one iteration.

    :param data_dict: This dictionary contains the following info

                        magnitude array      -> The numpy array containing the magnitude array
                        magnitude mask       -> The boolean mask for the magnitude array
                        magnitude mask not   -> The boolean mask for the magnitude array.
                                                This is simply
                                                        not*magnitude_mask
                        support              -> The boolean array for the support
                        diffraction      -> The diffraction field from previous step
                                                 Notice that this is not essential. I include it
                                                here only becase it can be useful for later usage.
                        density          -> The density from previous iteration
                        new diffraction flag -> Whether to update the new diffraction or not

    :param holder_dict:
                        This dictionary contains intermediate results to reduce memory allocation
                        time.

                        new diffraction with magnitude constrain  -> This is the diffraction field
                                                                     with magnitude constrain

                        new diffraction magnitude   -> This is the magnitude of the diffraction
                                                        field before applying the magnitude
                                                        constrain

                        new density tmp      -> This is the new density derived from the diffraction
                                                field with the magnitude constrain

                        phase holder          -> This is the PhaseTool of the derived diffraction
                                                    field

                        support not        -> np.logic_not(support)

                        modified support   -> In the algorithm, one needs to change pixels
                                                     in the support and in the same time
                                                     satisfied some conditions

                        support holder temporary        -> Sorry I just reall can not think up a name for
                                                this holder. It's used to store the information
                                                before one calculate the modified support

                        tmp holder 2        -> Sorry I just reall can not think up a name for
                                                this holder. This is for the approximated
                                                magnitude projection operator.
                        tmp holder 3        -> Sorry I just reall can not think up a name for
                                                this holder. This is for the approximated
                                                magnitude projection operator.
                        tmp holder 4        -> Sorry I just reall can not think up a name for
                                                this holder. This is for the approximated
                                                magnitude projection operator.

    :return: None. The resut is directly saved to the dictionary. This is also the very reason
            why I choose this mutable structure.
    """

    # Because this is a very basic function, I will not check paramters or counting the time

    # Get input variables
    mag = data_dict['magnitude array']
    mag_m = data_dict['magnitude mask']
    density = data_dict['density']

    # Get holder variables
    support_n = holder_dict['support not']

    ndiff_m = holder_dict['new diffraction magnitude']
    ndiff_c = holder_dict['new diffraction with magnitude constrain']

    phase = holder_dict['PhaseTool holder']

    # Step 1: Calculate the fourier transformation of the density
    ndiff_c[:] = np.fft.fftn(density)

    # Step 2: Apply magnitude constrain to the diffraction
    np.absolute(np.absolute(ndiff_c[mag_m]), out=ndiff_m[mag_m])
    np.divide(ndiff_c[mag_m], ndiff_m[mag_m],
              out=phase[mag_m], where=ndiff_m[mag_m] > 0)

    np.multiply(mag[mag_m], phase[mag_m], out=ndiff_c[mag_m])

    # Step 3: Get the updated density
    density[:] = np.fft.ifftn(ndiff_c).real

    # If it's out of the support then, set it to zero
    density[support_n] = 0.

    # If it's negative, then set it to zero
    density[density < 0] = 0

import numpy as np
from phase import util


def iterative_projection_normal(info_dict):
    """
    This function carries out the iterative projection for one iteration.

    :param info_dict: This is the dictionary containing all information essential for the
                        calculation. This dict should contain the following information

                        magnitude array      -> The numpy array containing the magnitude array
                        magnitude mask       -> The boolean mask for the magnitude array
                        magnitude mask not   -> The boolean mask for the magnitude array.
                                                This is simply
                                                        not*magnitude_mask

                        old diffraction    -> The diffraction field from previous step
                                                Notice that this is not essential. I include it
                                                here only becase it can be useful for later usage.
                        old density        -> The density from previous iteration
                        support              -> The boolean array for the support


                        new density          -> This is the new density derived from the diffraction
                                                field with the magnitude constrain

                        new density final   -> This is the new density that will be used

                        new diffraction with magnitude constrain  -> This is the diffraction field
                                                                     with magnitude constrain

                        new diffraction magnitude   -> This is the magnitude of the diffraction
                                                        field before applying the magnitude
                                                        constrain

                        new diffraction flag  -> Whether to new the diffraction or not
                        new diffraction       -> This is the diffraction field derived from the
                                                    old density distribution

                        phase holder          -> This is the phase of the derived diffraction
                                                    field

                        modified support   -> In the algorithm, one needs to change pixels
                                                     in the support and in the same time
                                                     satisfied some conditions

                        tmp holder 1        -> Sorry I just reall can not think up a name for
                                                this holder. It's used to store the information
                                                before one calculate the modified support

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

    # Get values from the dictionary
    mag = info_dict['magnitude array']
    mag_m = info_dict['magnitude mask']

    odens = info_dict['old density']
    support = info_dict['support']
    support_m = info_dict['modified support']

    ndens = info_dict['new density']
    ndens_f = info_dict['new density final']

    ndiff = info_dict['new diffraction']
    ndiff_m = info_dict['new diffraction magnitude']

    phase = info_dict['phase']
    ndiff_c = info_dict['new diffraction with magnitude constrain']

    tmp_1 = info_dict['tmp holder 1']

    a, b, c, d, e, f = [info_dict['par_a'],
                        info_dict['par_b'],
                        info_dict['par_c'],
                        info_dict['par_d'],
                        info_dict['par_e'],
                        info_dict['par_f']]

    # Step 1: Calculate the fourier transformation of the density
    ndiff_c[:] = np.fft.fftn(odens)

    if info_dict['new diffraction flag']:
        ndiff[:] = ndiff_c[:]

    # Step 2: Apply magnitude constrain to the diffraction
    np.absolute(np.absolute(ndiff_c[mag_m]), out=ndiff_m[mag_m])
    np.divide(ndiff_c[mag_m], ndiff_m[mag_m],
              out=phase[mag_m], where=ndiff_m[mag_m] > 0)

    np.multiply(mag[mag_m], phase[mag_m], out=ndiff_c[mag_m])

    # Step 3: Get the updated density
    ndens[:] = np.fft.ifftn(ndiff_c)

    # Step 4: Apply real space constrain
    # Get the positions where to modify
    support_m[:] = support[:]

    np.add(e * ndens[support], f * odens[support], out=tmp_1[support])
    np.greater(tmp_1[support], 0, out=support_m[support])

    np.add(c * ndens, d * odens, out=ndens_f)
    np.add(a * ndens[support_m], b * odens[support_m], out=ndens_f[support_m])


def iterative_projection_approximate(data_dict, holder_dict, param_dict):
    """
    This function carries out the iterative projection for one iteration.

    This function aims to be a stablized version of the previous function. According to Professor
    Russell Luke in the paper

        Relaxed averaged alternating reflections for diffraction imaging.

    It seems that the way one impose the magnitude constrain also introduces some instablity.
    Therefore, he has proposed to used an approximation to the magnitude constrain opeartor. This
    function implements that idea.

    In this function, the magnitude operator is not implemented in the naive way. Rather, if one
    would like to know the detail, then he should either look that the paper and find the formula
    (34) or look at the function : approximate_magnitude_projection   in this script.

    However one should notice that, that approximation is not directly available in real world
    applications. The reason is complicated.

    The most important reason is that in that formula, Prof. Luke seems to have assumed that
    there is not gaps on the detectors. However, there are a lot of gaps on the detector. Therefore
    there will be a lot of edges in the magnitude array if one set the unknown pixels to be zero.
    However, these artificial edges will have huge influences on inverse fourier transformations.

    Therefore, I have added the following operations.
    1. Get the magnitude of the diffraction patterns derived from the input density operators.
    2. Replace the corresponding values in this pattern with the experiment values.
    3. Use the modified array to do the magnitude projection.


    :param data_dict: This dictionary contains the following info


                        magnitude array      -> The numpy array containing the magnitude array
                        magnitude mask       -> The boolean mask for the magnitude array
                        magnitude mask not   -> The boolean mask for the magnitude array.
                        This is
                                                simply
                                                        not*magnitude_mask

                        old diffraction    -> The diffraction field from previous step
                                                Notice that this is not essential. I include it
                                                here only becase it can be useful for later usage.
                        old density        -> The density from previous iteration
                        support              -> The boolean array for the support


                        new density          -> This is the new density derived from the diffraction
                                                field with the magnitude constrain

                        new density final   -> This is the new density that will be used

                        new diffraction with magnitude constrain  -> This is the diffraction field
                                                                     with magnitude constrain

                        new diffraction magnitude   -> This is the magnitude of the diffraction
                                                        field before applying the magnitude
                                                        constrain

                        new diffraction flag  -> Whether to new the diffraction or not
                        new diffraction       -> This is the diffraction field derived from the
                                                    old density distribution

                        phase holder          -> This is the phase of the derived diffraction
                                                    field

                        modified support   -> In the algorithm, one needs to change pixels
                                                     in the support and in the same time
                                                     satisfied some conditions

                        tmp holder 1        -> Sorry I just reall can not think up a name for
                                                this holder. It's used to store the information
                                                before one calculate the modified support

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

    # Get values from the dictionary
    mag = info_dict['magnitude array']
    mag_m = info_dict['magnitude mask']
    mag_mn = info_dict['magnitude mask not']

    odens = info_dict['old density']
    support = info_dict['support']
    support_m = info_dict['modified support']

    ndens = info_dict['new density']
    ndens_f = info_dict['new density final']

    ndiff = info_dict['new diffraction']
    ndiff_m = info_dict['new diffraction magnitude']

    phase = info_dict['phase']
    ndiff_c = info_dict['new diffraction with magnitude constrain']

    tmp_1 = info_dict['tmp holder 1']

    a, b, c, d, e, f = [info_dict['par_a'],
                        info_dict['par_b'],
                        info_dict['par_c'],
                        info_dict['par_d'],
                        info_dict['par_e'],
                        info_dict['par_f']]

    # Step 1: Calculate the fourier transformation of the density
    ndiff_c[:] = np.fft.fftn(odens)

    if info_dict['new diffraction flag']:
        ndiff[:] = ndiff_c[:]

    # Step 2: Apply magnitude constrain to the diffraction

    # Padding the gaps with the derived magnitude
    np.absolute(np.absolute(ndiff_c), out=ndiff_m)
    mag[mag_mn] = ndiff_m[mag_mn]

    # Apply the approximated operator
    approximate_magnitude_projection(diff=)

    # Step 3: Get the updated density
    ndens[:] = np.fft.ifftn(ndiff_c)

    # Step 4: Apply real space constrain
    # Get the positions where to modify
    support_m[:] = support[:]

    np.add(e * ndens[support], f * odens[support], out=tmp_1[support])
    np.greater(tmp_1[support], 0, out=support_m[support])

    np.add(c * ndens, d * odens, out=ndens_f)
    np.add(a * ndens[support_m], b * odens[support_m], out=ndens_f[support_m])


def approximate_magnitude_projection(diff, _holder_1, _holder_2, _holder_3, mag, epsilon):
    """
       This is a new operator to replace the original magnitude operator. According to professor
    Luke in the paper

        http://iopscience.iop.org/article/10.1088/0266-5611/21/1/004

        Relaxed averaged alternating reflections for diffraction imaging

    This new operator is more stable. Concrete formula is (34) in the paper. Notice that the
    formula contains a typo. I here represents the identity operator and should be replaced by u.


    :param diff: The estimated diffraction field with phase
    :param _holder_1: The holder for intermediate calculation
    :param _holder_2:
    :param _holder_3:
    :param mag: The magnitude array. Notice that, here, the magnitude array might not be the
                original one. Because the original array have edges if one simply assign zeros to
                the missing data, one might consider to assign values from the estimation to reduce
                the artifical edges.
    :param epsilon: A epsilon value used to calculate the true epsilon value. The detail should be
                    find from the article.
    :return:
    """
    # Get the fourier transform
    _holder_1 = np.fft.fftn(diff)

    # Get the norm of the transformed data
    _holder_2 = util.abs2(_holder_1)

    # Calculate the true epsilon that should be used in the calculation
    teps = (epsilon * np.max(_holder_2)) ** 2

    # Calculatet the output without truely return any array
    _holder_3 = np.divide(np.multiply(_holder_2 - np.multiply(mag, np.sqrt(_holder_2 + teps)),
                                      np.multiply(_holder_2 + 2 * teps, _holder_1)),
                          np.square(_holder_2 + teps))
    return diff - np.fft.ifftn(_holder_3)

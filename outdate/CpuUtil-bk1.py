import numpy as np
from PhaseTool import util


def iterative_projection_normal(data_dict, holder_dict, a, b, c, d, e, f):
    """
    This function carries out the iterative projection for one iteration.

    :param data_dict:
                        "magnitude array": self.magnitude,
                        "magnitude mask": self.magnitude_mask,
                        "support": self.support,
                        "diffraction": self.diffraction,
                        "density": self.density

    :param holder_dict:

        "magnitude mask not": np.logical_not(self.magnitude_mask),
        "support not": np.logical_not(self.support),


        "diffraction": self.diffraction,           -> The fourier of the initial density
        "new diffraction magnitude":               -> np.abs(diffraction)
        "phase holder":                            -> The phase of the diffraction
        "diffraction with magnitude constrain":    -> Diffraction with the magnitude constrain


        "new density tmp":                         -> Real part of the fourier trans of the
                                                      diffraction with magnitude constrain

        "support holder temporary":                -> Holder when getting new support for update
        "modified support":                        -> Support combined with Algorithm constrain


        "tmp holder 2": dtype=np.complex128,       -> For approximated projection
        "tmp holder 3": dtype=np.float64           -> For approximated projection
        "tmp holder 4": dtype=np.complex128        -> For approximated projection

    :param a:
    :param b:
    :param c:
    :param d:
    :param e:
    :param f:
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
    ndiff_c = holder_dict['diffraction with magnitude constrain']
    phase = holder_dict['phase holder']

    # Step 1: Calculate the fourier transformation of the density
    ndiff_c[:] = np.fft.fftn(density)

    # Step 2: Apply magnitude constrain to the diffraction
    # np.absolute(ndiff_c[mag_m], out=ndiff_m[mag_m].real)
    # print(np.sum(np.abs(ndens_t)))

    # np.divide(ndiff_c[mag_m], ndiff_m[mag_m], out=phase[mag_m], where=ndiff_m[mag_m] > 0)
    # np.divide(ndiff_c[mag_m], ndiff_m[mag_m], out=phase[mag_m])

    # np.multiply(mag[mag_m], phase[mag_m], out=ndiff_c[mag_m])

    phase[:] = util.get_phase(ndiff_c)

    # Step 3: Get the updated density
    ndens_t[:] = np.fft.ifftn(ndiff_c).real

    # Step 4: Apply real space constrain
    # Get the positions where to modify
    support_m[:] = support[:]

    # np.add(e * ndens_t[support], f * density[support], out=support_t[support])
    support_t[support] = e * ndens_t[support] + f * density[support]
    np.greater(support_t[support], 0, out=support_m[support])

    # Update the modified support not
    np.logical_not(support_m, out=support_mn)

    # Apply the real space update rule
    # np.add(c * ndens_t[support_mn], d * density[support_mn], out=density[support_mn])
    # np.add(a * ndens_t[support_m], b * density[support_m], out=density[support_m])
    density[support_mn] = c * ndens_t[support_mn] + d * density[support_mn]
    density[support_m] = a * ndens_t[support_m] + b * density[support_m]

    # print(np.sum(np.abs(density - data_dict['density'])))


def error_reduction(data_dict, holder_dict):
    """
    This function carries out the iterative projection for one iteration.

    :param data_dict:
                        "magnitude array": self.magnitude,
                        "magnitude mask": self.magnitude_mask,
                        "support": self.support,
                        "diffraction": self.diffraction,
                        "density": self.density

    :param holder_dict:

        "magnitude mask not": np.logical_not(self.magnitude_mask),
        "support not": np.logical_not(self.support),


        "diffraction": self.diffraction,           -> The fourier of the initial density
        "new diffraction magnitude":               -> np.abs(diffraction)
        "phase holder":                            -> The phase of the diffraction
        "diffraction with magnitude constrain":    -> Diffraction with the magnitude constrain


        "new density tmp":                         -> Real part of the fourier trans of the
                                                      diffraction with magnitude constrain

        "support holder temporary":                -> Holder when getting new support for update
        "modified support":                        -> Support combined with Algorithm constrain


        "tmp holder 2": dtype=np.complex128,       -> For approximated projection
        "tmp holder 3": dtype=np.float64           -> For approximated projection
        "tmp holder 4": dtype=np.complex128        -> For approximated projection

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
    ndiff_c = holder_dict['diffraction with magnitude constrain']

    phase = holder_dict['phase holder']

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

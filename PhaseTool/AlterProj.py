import numpy as np
import copy
from PhaseTool import util
from PhaseTool import CpuUtil

"""
This is the main interface to the applications.

The basic idea is that, the use can create a object to do phase retrieval.
The reason that I would like to use class is that after a short discussion with Zhen,
I realized that class makes it easier for the user to use. Also, in the algorithm, to
accelerate the calculation, I need to initialize a lot of variables before the
calculation. Therefore, it would be very clumsy to use dictionary or the other
methods for the users to use.
"""


class CpuAlterProj:
    def __init__(self):
        """
        One can use this method to initialize a CDI object and can create a newer one
        with information from this object.
        """

        # Meta parameters
        size = 64
        # History recorder
        self.history = []

        ########################
        # Data Parameters
        ########################
        # The dimension of the data
        self.dim = 2

        # The shape of the data in this problem
        self.data_shape = (size, size)

        # The position of the center of the diffraction measured in pixel
        self.center_in_pixel = np.array([32., 32.], dtype=np.float64)

        # IO variables
        self.magnitude = np.zeros((size, size), dtype=np.float64)
        self.magnitude_mask = np.ones((size, size), dtype=np.bool)
        self.support = np.ones((size, size), np.bool)
        self.density = np.ones((size, size))

        ########################
        # Important holder
        ########################
        self.diffraction = np.ones((size, size), dtype=np.complex128)

        ########################
        # Shrink Wrap Parameters
        ########################
        self.shrink_wrap_on = False
        self.shrink_wrap_fill_holes = False
        self.shrink_wrap_convex_hull = False
        self.shrink_wrap_counter = 0

        self.shrink_wrap_rate = 30  # The number of iterations before the shrink wrap occurs.
        self.shrink_wrap_sigma = 5.  # The sigma in the gaussian filter in the process.

        # The ratio the sigma decays after each application
        self.shrink_wrap_sigma_decay_ratio = 0.95

        self.shrink_wrap_threshold_retio = 0.04
        self.shrink_wrap_threshold_decay_ratio = 1.0

        self.shrink_wrap_hist = {'threshold rate history': [],
                                 'sigma history': []}

        self.shrink_wrap_keep_history = False

        ########################
        # Alternating Projection parameters
        ########################

        self.available_algorithms = ['ER', 'HIO', 'HPR', 'RAAR', 'GIF-RAAR']
        self.algorithm = "RAAR"
        """
        In the following dictionary

        epsilon :               -> parameter for the approximation

        [par_a par_b par_c par_d par_e par_f] are complicated. They defines the structure of the
        projections.

        Below, I use u_(n+1) to denote the new density, then in general,
        the algorithm can be represented as

                     par_c * P(u_n)(x) + par_d * u_n(x)
        u_(n+1)(x) =
                     par_a * P(u_n)(x) + par_b * u_n(x)     for x (in support) and
                                                            par_e * P(u_n)(x) >
                                                            par_f * u_n(x)
        """
        self.param_dict = {"par_a": 0.,
                           "par_b": 0.,
                           "par_c": 0.,
                           "par_d": 0.,
                           "par_e": 0.,
                           "par_f": 0.,
                           "epsilon": 1e-15}

        self.iter_num = 1200
        self.beta = 0.75 * np.ones(self.iter_num, dtype=np.float64)
        self.beta_decay = True
        self.beta_decay_rate = 20

        ########################
        # Dictionaries
        ########################
        """
        The following two dictionaries are the core of this class. The data class contains all 
        the external IO information and it used to facilitate the IO.
        
        The data_dict dictionary contains five items
        
        "magnitude array": self.magnitude,
        "magnitude mask": self.magnitude_mask,
        "support": self.support,
        "diffraction": self.diffraction,
        "density": self.density
        
        They are respectively the corresponding properties of this class. Their meaning is 
        obvious, therefore, I'll not give more explanation.
        
        
        The holder_dict contains variables to reduce the memory allocation time. It contains the 
        following entries.
        
        
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
        "modified support not":                    -> np.logical_not(modified support)


        "tmp holder 2": dtype=np.complex128,       -> For approximated projection
        "tmp holder 3": dtype=np.float64           -> For approximated projection
        "tmp holder 4": dtype=np.complex128        -> For approximated projection
        """
        self.data_dict = {"magnitude array": self.magnitude,
                          "magnitude mask": self.magnitude_mask,
                          "support": self.support,
                          "density": self.density}

        self.holder_dict = {"magnitude mask not": np.logical_not(self.magnitude_mask),
                            "support not": np.logical_not(self.support),

                            "diffraction": self.diffraction,
                            "new diffraction magnitude": np.zeros((size, size), dtype=np.float64),
                            "phase holder": np.zeros((size, size), dtype=np.float64),
                            "diffraction with magnitude constrain": np.zeros((size, size),
                                                                             dtype=np.complex128),

                            "new density tmp": np.zeros((size, size), dtype=np.float64),

                            "modified support": np.zeros((size, size), dtype=np.bool),
                            "modified support not": np.ones((size, size), dtype=np.bool),
                            "support holder temporary": np.zeros((size, size), dtype=np.float64),

                            "tmp holder 2": np.zeros((size, size), dtype=np.complex128),
                            "tmp holder 3": np.zeros((size, size), dtype=np.float64),
                            "tmp holder 4": np.zeros((size, size), dtype=np.complex128)
                            }

    ################################################################################################
    # Initialize and update
    ################################################################################################
    def initialize_easy(self, magnitude, magnitude_mask, full_initialization=True):
        """
        Give a minimal initialization of the data properties.

        :param magnitude: The magnitude of the experiment
        :param magnitude_mask: The mask for the detector.
        :param full_initialization: This is a flag. If this is set to be true. Then this function
                                    initialize all properties according to the magnitude and the
                                    magnitude_mask assuming that

                                    1. Use a random phase
                                    2. Calculate the density from the diffraction with random phase
                                    3. Drive a support with auto-correlation function
                                    4. Use default RAAR algorithm with default decaying beta values
                                        and default iteration number
                                    5. Use shrink wrap algorithm to improve the support estimation
                                    6. Calculate default metrics to measure the convergence of the
                                        algorithm. Especially, the error is calculated with respect
                                        to the input magnitude array.
                                    7. Save the data in the default address with default name in
                                        the default format.

                                    If this is set to be false, this function will only make some
                                    self-consistent changes. The user still has to initialize
                                    the calculation parameters, the initialization condition
        :return:
        """
        self.magnitude = np.fft.fftshift(magnitude)
        self.magnitude_mask = np.fft.fftshift(magnitude_mask)

        # Execute later-on initialization
        if full_initialization:
            self._full_init_from_initez()
        else:
            self._init_from_initez()

    def _init_from_initez(self):
        """
        This is an internal method used to initialize some more properties when the user has used
        the  method minimal_data_initialize(magnitude, magnitude_mask).

        This method only initializes some not very important variables. The user has to initialize
        the calculation parameters, the initial values and the other stuff.
        :return:
        """

        self.data_dict["magnitude array"] = self.magnitude
        self.data_dict["magnitude mask"] = self.magnitude_mask
        self.holder_dict["magnitude mask not"] = np.logical_not(self.magnitude_mask)

        # Abstract info
        self.data_shape = copy.deepcopy(self.magnitude.shape)
        self.dim = len(self.magnitude.shape)

        # Set and check the origin of the origin of the data
        self.set_and_check_detector_origin()

    def _full_init_from_initez(self):
        """
        This is an internal method used to initialize some more properties when the user has
        used the method minimal_data_initialize(magnitude, magnitude_mask).

        In short, this function initialize all

        :return:
        """

        # TODO: This default behavior should be modified
        # Step 1: Finish the most elementary initialization
        self._init_from_initez()

        # Step 2: Initialize the support
        self.use_auto_support()
        print("Initialize the support array with the auto-correlation array using "
              "default methods with default parameters.")

        # Step 3: Set the initial diffraction and initial density values
        self.derive_initial_density(fill_detector_gap=True, method="Random")

        # Step 4: Set the Shrink Wrap Properties
        self.shrink_warp_properties(
            on=self.shrink_wrap_on,
            threshold_ratio=self.shrink_wrap_threshold_retio,
            sigma=self.shrink_wrap_sigma,
            decay_rate=self.shrink_wrap_rate,
            threshold_ratio_decay_ratio=self.shrink_wrap_threshold_decay_ratio,
            sigma_decay_ratio=self.shrink_wrap_sigma_decay_ratio,
            filling_holes=self.shrink_wrap_fill_holes,
            convex_hull=self.shrink_wrap_convex_hull)

        # Step 5: Set beta and iteration number
        self.set_beta_and_iter_num(
            beta=self.beta,
            iter_num=self.iter_num,
            decay=self.beta_decay,
            decay_rate=self.beta_decay_rate)

        # Step 6: Set the Algorithm
        self.set_algorithm(alg_name=self.algorithm)

        # Step 4: Initialize the holder variables.
        print("Update the input and holder dictionaries")
        self.update_input_dict()
        self.update_holder_dict()

    def use_auto_support(self, threshold=0.04, gaussian_filter=True,
                         sigma=1.0, fill_detector_gap=True, bin_num=300):
        """
        Generate a support from the autocorrelation function calculated from the support info.

        By default, this function will calculate the autocorrelation from the self magnitude array.
        However, because there are some gaps in the detector, it can be desirable to fill those
        gaps before calculating the autocorrelation.

        :param threshold: The threshold that is used to decided whether the specific pixel is
                            in the support or not. Specifically,

                            span = max(auto) - min(auto)
                            if val(pixel) >= threshold * span + min(auto):
                                sup(pixel) = True
        :param gaussian_filter: Where to apply the guassian field after one obtains the
                                auto correlation. By default, one should use this filter.
        :param sigma: The sigma value used in the Gaussian filter.
        :param fill_detector_gap: Whether to fill those gaps in the detector.
        :param bin_num: This is the bin number used to get radial information. This only has effect
                        when the fill_detector_gap is set to be True. The detail should be obvious
                        from the code.
        :return:
        """

        self.support = util.get_support_from_autocorrelation(
            magnitude=self.magnitude,
            magnitude_mask=self.magnitude_mask,
            origin=self.center_in_pixel,
            threshold=threshold,
            gaussian_filter=gaussian_filter,
            gaussian_sigma=sigma,
            flag_fill_detector_gap=fill_detector_gap,
            bin_num=bin_num
        )

        # Update the input dictionary
        self.data_dict["support"] = self.support
        self.holder_dict["support not"] = np.logical_not(self.support)

        return self.support

    def set_support(self, support):
        """

        :param support:
        :return:
        """
        self.support = support

        # Update the input dictionary.
        self.data_dict["support"] = self.support
        self.holder_dict["support not"] = np.logical_not(self.support)

    def set_initial_density(self, density):
        """
        Set the initial density and update the input dictionary for the algorithm.

        :param density:
        :return:
        """
        self.density = density
        self.data_dict["density"] = self.density
        print("The initial density is updated.")

    def set_beta_and_iter_num(self, iter_num, beta=None, decay=False, decay_rate=20):
        """
        This function set the beta for the algorithm. In this algorithm, for simplicity, the
        iteration number has to be the same as the number of betas.

        In this function, beta can be either a scalar number, or a tuple, list, 1-d numpy array.
        This function all automatically convert them in to proper format.

        If beta is a scalar, then
            self.beta = beta * np.ones(self.iter_num)

        otherwise
            self.beta = np.array(beta)
            self.iter_num = self.beta.shape[0]

        :param beta:
        :param iter_num:
        :param decay:
        :param decay_rate:
        :return:
        """
        try:
            iter_num = int(iter_num)
        except ValueError:
            raise Exception("The iter_num argument has to be a positive integer.")

        # Step 1: Check if the beta value is a list.
        if isinstance(beta, (list, tuple, np.ndarray)):

            """
            Case 1: The user specifies the beta values completely. Then ignore the other two 
                    arguments and derive the iteration number from the beta values.
            """

            self.beta = np.array(beta)
            self.iter_num = self.beta.shape[0]
            print("The value of the beta argument is a {}".format(type(beta)))
            print("Therefore, the 'decay' and 'iter_num' arguments are ignored.")
            print("The iteration number is set to be the length of the beta array which "
                  "currently is {}".format(beta.shape[0]))

        # Step 2: Check if beta is a scalar.
        elif isinstance(beta, (int, float, complex)):

            """
            Case 2: The user only gives a constant number for beta
            
            """
            if decay:
                """
                If decay == True, then create a decaying sequence.
                """
                self._use_decaying_beta(initial_beta=beta,
                                        decay_rate=decay_rate,
                                        iter_num=iter_num)
                print("The user uses a constant value for the beta value. This values"
                      "beta = {}".format(beta))
                print("This values is recognized as the initial beta value for a list of "
                      "decaying beta values. The length of this list is the iter_num value"
                      "which is {}".format(iter_num))

            else:
                """
                If decay == False, then create an array which is consists of the same values.
                """
                self.beta = beta * np.ones(iter_num)
                self.iter_num = iter_num
                print("The user uses a constant value for the beta argument. The argument "
                      "decay is set to False. The iteration number is {}".format(iter_num))

        # Step 3: Beta is not specified.
        else:

            print("Since the argument beta is not specified, the arguments decay and decay_rate "
                  "are ignored.")

            if self.beta:

                # If the existing beta is a scalar, then create a new list of beta with this value
                #
                if isinstance(self.beta, (int, float, complex)):
                    self.beta = self.beta * np.ones(iter_num, dtype=np.float64)

                # If the existing beta is a list, then take sublist or extend it to the current
                # iteration number
                elif isinstance(self.beta, (list, tuple, np.ndarray)):
                    self.beta = np.array(self.beta)
                    if self.beta.shape[0] >= iter_num:
                        self.beta = self.beta[:iter_num]
                        self.iter_num = iter_num

                    else:
                        tmp = np.copy(self.beta)
                        tmp_2 = tmp.shape[0]

                        # Update the two properties
                        self.beta = np.zeros(iter_num, dtype=np.float64)
                        self.iter_num = iter_num

                        # Initialize the new beta array
                        self.beta[:tmp_2] = tmp[:]
                        self.beta[tmp_2:] = tmp[-1]

            else:
                """
                In this case, the user has set none value to the self.beta property for whatever 
                reason. And in the same time, the user does not provide a new valid beta value 
                In this case, whether it's using ER or any other algorithm, just set beta to be 
                ones. 
                """
                # Due to some unexpected reason,
                self.beta = np.ones(iter_num, dtype=np.float64)
                self.iter_num = iter_num

    def _use_decaying_beta(self, initial_beta=0.75, decay_rate=7, iter_num=200):
        """
        According to the paper

        Relaxed averaged alternating reflections for diffraction imaging

        The following beta seems to be useful.

        beta_n = beta_0 + (1 - beta_0) * (1 - exp( - (n/7)**3))

        :param iter_num:
        :param decay_rate:
        :param initial_beta:
        :return:
        """

        beta_0 = initial_beta
        self.iter_num = iter_num

        tmp_list = np.divide(np.arange(self.iter_num, dtype=np.float64), decay_rate)
        tmp_list = np.multiply(1 - np.exp(-np.power(tmp_list, 3)), 1 - beta_0)
        self.beta = beta_0 + tmp_list

    def set_algorithm(self, alg_name):
        """

        :param alg_name:
        :return:
        """

        if alg_name in self.available_algorithms:
            self.algorithm = alg_name

            # Initialize the parameter dictionary
            if self.algorithm == 'HIO':
                self.param_dict["par_a"] = 1.
                self.param_dict["par_b"] = 0.
                self.param_dict["par_c"] = - 0.87
                self.param_dict["par_d"] = 1.
                self.param_dict["par_e"] = 1.
                self.param_dict["par_f"] = 0.
            elif self.algorithm == 'ER':
                self.param_dict["par_a"] = 1.
                self.param_dict["par_b"] = 0.
                self.param_dict["par_c"] = 0.
                self.param_dict["par_d"] = 0.
                self.param_dict["par_e"] = 1.
                self.param_dict["par_f"] = 0.
            elif self.algorithm == 'HPR':
                self.param_dict["par_a"] = 1.
                self.param_dict["par_b"] = 0.
                self.param_dict["par_c"] = - 0.87
                self.param_dict["par_d"] = 1.
                self.param_dict["par_e"] = 1.87
                self.param_dict["par_f"] = -1.
            elif self.algorithm == "RAAR":
                self.param_dict["par_a"] = 1.
                self.param_dict["par_b"] = 0.
                self.param_dict["par_c"] = 1 - 2 * 0.87
                self.param_dict["par_d"] = 0.87
                self.param_dict["par_e"] = 2.
                self.param_dict["par_f"] = -1.
            elif self.algorithm == "GIF-RAAR":
                self.param_dict["par_a"] = 1.87
                self.param_dict["par_b"] = - 0.87
                self.param_dict["par_c"] = 1 - 2 * 0.87
                self.param_dict["par_d"] = 0.87
                self.param_dict["par_e"] = 3.
                self.param_dict["par_f"] = -1.

        else:

            print("Currently, only the following algorithms are available.")
            print(self.available_algorithms)
            raise Exception("Please have a look at the info above.")

    def update_param_dict_with_beta(self, beta):
        if self.algorithm == "HIO":
            self.param_dict["par_c"] = - beta
        elif self.algorithm == "HPR":
            self.param_dict["par_c"] = - beta
            self.param_dict["par_e"] = 1 + beta
        elif self.algorithm == "RAAR":
            self.param_dict["par_c"] = 1 - 2 * beta
            self.param_dict["par_d"] = beta
        elif self.algorithm == "GIF-HPR":
            self.param_dict["par_a"] = 1 + beta
            self.param_dict["par_b"] = - beta
            self.param_dict["par_c"] = 1 - 2 * beta
            self.param_dict["par_d"] = beta
        else:
            raise Exception("Algorithm {} is not available".format(self.algorithm) +
                            " at present. Please set the " +
                            "self.algorithm propoerty to be one of the following values:\n" +
                            "{} with function".format(self.available_algorithms) +
                            "self.set_algorithm.")

    def execute_algorithm(self, algorithm=None, beta=None, iter_num=None, beta_decay=None,
                          beta_decay_rate=None, initial_density=None, shrink_wrap_on=None):
        """

        :param algorithm:
        :param beta:
        :param iter_num:
        :param beta_decay:
        :param beta_decay_rate:
        :param initial_density:
        :param shrink_wrap_on:

        :return:
        """

        # Set Algorithem
        if algorithm:
            self.set_algorithm(alg_name=algorithm)

        # Set beta
        if beta or iter_num:
            self.set_beta_and_iter_num(beta=beta, iter_num=iter_num,
                                       decay=beta_decay,
                                       decay_rate=beta_decay_rate)

        # Set initial density
        if initial_density:
            self.set_initial_density(density=initial_density)

        # Set the shrink wrap behavior
        if type(shrink_wrap_on) is bool:
            self.shrink_wrap_on = shrink_wrap_on
            if shrink_wrap_on:
                print("Enable shrink wrap function.")

        # Prepare for the calculation
        if not (self.algorithm in self.available_algorithms):

            # If the algorithm is not available, give an error.
            raise Exception("There is something wrong with the self.algorithm property." +
                            "At present, the only available algorithms are " +
                            "{}".format(self.available_algorithms))

        # If the algorithm is available, check self-consistency and do the calculation
        else:
            print("Using algorithm {}".format(self.algorithm))

            # Update the input dict and the holder dict
            self.update_input_dict()
            self.update_holder_dict()

            # Check self consistency
            self.check_consistency_short()
            print("Finishes self-consistency check.")

            # Begin
            if self.algorithm == "ER":

                for idx in range(self.iter_num):

                    # Step 1: Execute the ER
                    CpuUtil.error_reduction(data_dict=self.data_dict,
                                            holder_dict=self.holder_dict)

                    # Step 2: Check if one should apply the shrink wrap
                    if self.shrink_wrap_on:
                        if np.mod(idx, self.shrink_wrap_rate) == 0:
                            tmp = util.shrink_wrap(
                                density=self.data_dict["density"],
                                sigma=self.shrink_wrap_sigma,
                                threshold_ratio=self.shrink_wrap_threshold_retio,
                                filling_holds=self.shrink_wrap_fill_holes,
                                convex_hull=self.shrink_wrap_convex_hull)

                            self.set_support(support=tmp)
                            self.update_shrink_wrap_properties()

            else:

                for idx in range(self.iter_num):

                    # Step 1: Execute the alternating projections
                    CpuUtil.iterative_projection_normal(data_dict=self.data_dict,
                                                        holder_dict=self.holder_dict,
                                                        a=self.param_dict["par_a"],
                                                        b=self.param_dict["par_b"],
                                                        c=self.param_dict["par_c"],
                                                        d=self.param_dict["par_d"],
                                                        e=self.param_dict["par_e"],
                                                        f=self.param_dict["par_f"])

                    # Step 3: Update the beta parameter
                    self.update_param_dict_with_beta(beta=self.beta[idx])

                    # Step 4: Check if one should apply the shrink wrap
                    if self.shrink_wrap_on:
                        if np.mod(idx, self.shrink_wrap_rate) == 0:
                            print("This is iteration No.{}".format(idx))
                            print("Update the support with shrink wrap.")

                            tmp = util.shrink_wrap(
                                density=self.data_dict["density"],
                                sigma=self.shrink_wrap_sigma,
                                threshold_ratio=self.shrink_wrap_threshold_retio,
                                filling_holds=self.shrink_wrap_fill_holes,
                                convex_hull=self.shrink_wrap_convex_hull)

                            self.set_support(support=tmp)
                            self.update_shrink_wrap_properties()

    def shrink_warp_properties(self, on=False, threshold_ratio=0.04, sigma=5.0, decay_rate=30,
                               threshold_ratio_decay_ratio=1.0, sigma_decay_ratio=0.95,
                               filling_holes=True, convex_hull=False, keep_history=True):
        """
        This is just a temporary implementation of the shrink warp tuning function.
        There are actually, quite a lot of parameters to tune and that has to be consistent with 
        the total iteration number. Therefore, it's complicated. 
        
        So at, present, I will do the folloing things. Instead of getting an array with 

        :param on:
        :param threshold_ratio: 
        :param sigma: 
        :param decay_rate: 
        :param threshold_ratio_decay_ratio: 
        :param sigma_decay_ratio: 
        :param filling_holes:
        :param convex_hull:
        :param keep_history:
        :return: 
        """
        self.shrink_wrap_on = on
        if on:
            print("Enable shrink wrap functions.")
        else:
            print("Disable shrink wrap functions.")
            return

        if threshold_ratio:
            self.shrink_wrap_threshold_retio = threshold_ratio
            print("The initial threshold ratio of the shrink warp algorithm is set to {}".format(
                threshold_ratio))

        if sigma:
            self.shrink_wrap_sigma = sigma
            print("The initial sigma of the shrink warp algorithm is set to {}".format(sigma))

        if decay_rate:
            self.shrink_wrap_rate = int(decay_rate)
            print("The decay_rate argument is set to be {}".format(decay_rate))
            print("Therefore, the shrink wrap algorithm will be applied every {}".format(
                decay_rate) +
                  "iterations of the projections. The change of the parameters of the shrink" +
                  " wrap function will occur after each application. Therefore, if you would like" +
                  " to use a constant parameter for all shrink wraps, please set the " +
                  "argument threshold_ratio_decay_ratio=1.0, and "
                  "sigma_decay_ratio=1.0. By default, they are 0.9.")

        if threshold_ratio_decay_ratio:
            self.shrink_wrap_threshold_decay_ratio = threshold_ratio_decay_ratio
            print("The threshold ratio will decay "
                  "to {} * (current ratio)".format(threshold_ratio_decay_ratio) +
                  "after each application of the shrink wrap algorithm. To stop this decaying,"
                  "please set threshold_ratio_decay_ratio=1. when calling this funciton. ")

        if sigma_decay_ratio:
            self.shrink_wrap_sigma_decay_ratio = sigma_decay_ratio
            print("The sigma will decay "
                  "to {} * (current ratio)".format(sigma_decay_ratio) +
                  "after each application of the shrink wrap algorithm. To stop this decaying,"
                  "please set threshold_ratio_decay_ratio=1. when calling this funciton. ")

        if type(filling_holes) == bool:
            if filling_holes:
                self.shrink_wrap_fill_holes = True
                print("As per request, when update the support, fill holes in the support "
                      "derived from standard shrink wrap algorithm.")
            else:
                self.shrink_wrap_fill_holes = False

        if type(convex_hull) == bool:
            if convex_hull:
                self.shrink_wrap_convex_hull = True
                print("As per request, when update the support, use the convex hull of the result "
                      "of the standard shrink wrap algorithm as the suuport.")
            else:
                self.shrink_wrap_convex_hull = False

        if type(filling_holes) == bool and type(convex_hull) == bool:
            if (not filling_holes) and (not convex_hull):
                print("When update the support, use the standard result of the shrink wrap "
                      "algorithm as the new support.")

        self.shrink_wrap_keep_history = keep_history

    def update_shrink_wrap_properties(self, keep_history=False):
        """
        Update some properties and keep track of the history.

        :return:
        """
        if keep_history:
            self.shrink_wrap_hist['threshold rate history'].append(self.shrink_wrap_threshold_retio)
            self.shrink_wrap_hist['sigma history'].append(self.shrink_wrap_sigma)

        self.shrink_wrap_sigma *= self.shrink_wrap_sigma_decay_ratio
        self.shrink_wrap_threshold_retio *= self.shrink_wrap_threshold_decay_ratio

        self.shrink_wrap_counter += 1

    ################################################################################################
    # Details
    ################################################################################################
    ###################################
    # Initialization details
    ###################################

    def derive_initial_density(self, fill_detector_gap=True, method="Random"):

        # Step 1: Get the phase
        if method in ('Random', 'Zero', 'Minimal', 'Support'):
            if method == "Random":

                # Create a central symmetric phase array
                tmp1 = np.random.rand(*self.magnitude.shape)
                tmp2 = np.copy(tmp1)
                for l in range(self.dim):
                    tmp2 = np.flip(m=tmp2, axis=l)

                phase_array = np.exp(1j * np.pi * (tmp1 - tmp2))

            elif method == "Zero":
                phase_array = np.ones_like(self.magnitude)
            elif method == "Support":
                print("Using the support as the initial guess of the density.")
                phase_array = np.zeros_like(self.support, dtype=np.float64)
                phase_array[self.support] = 1.

            else:
                raise Exception("Sorry, the minimal phase initialization method is not "
                                "implemented yet")
        else:
            raise Exception("At present, the phase can only be 'Random', 'Zero' or 'Minimal'.")

        # Step 2: Fix the detector gaps
        if fill_detector_gap:
            magnitude_tmp = self.fill_detector_gaps(gaussian_filter=True,
                                                    gaussian_sigma=1.0,
                                                    bin_num=300)
        else:
            magnitude_tmp = self.magnitude

        # Step 3: Get all the values
        self.diffraction = np.multiply(phase_array, magnitude_tmp)

        if method in ('Random', 'Zero', 'Minimal'):
            self.density = np.fft.ifftn(self.diffraction).real

        else:
            self.density = np.copy(phase_array)

    ###################################
    # Parameter & consistency check
    ###################################
    def check_consistency_short(self):
        """

        :return:
        """
        tmp = [self.data_shape,
               self.magnitude.shape,
               self.magnitude_mask.shape,
               self.support.shape]

        if len(set(tmp)) != 1:
            print("[self.data_shape,self.magnitude.shape,"
                  "self.magnitude_mask.shape,self.support.shape]")
            print(tmp)
            raise Exception("The data shape are not the same for the above 4 variables."
                            "Please initialize them properly.")

    def check_self_consistency(self):
        """
        This function check if all the properties in this object is consistent with each other.
        :return:
        """

        # check detector center
        self.set_and_check_detector_origin()

        # Check the other properties of self

        shape_list = [self.data_shape,
                      self.magnitude.shape,
                      self.magnitude_mask.shape,
                      self.diffraction.shape,
                      self.density.shape,
                      self.support.shape,

                      self.data_dict["magnitude array"].shape,
                      self.data_dict["magnitude mask"].shape,
                      self.data_dict["support"].shape,
                      self.data_dict["density"].shape,

                      self.holder_dict["magnitude mask not"].shape,
                      self.holder_dict["support not"].shape,

                      self.holder_dict["diffraction"].shape,
                      self.holder_dict["diffraction with magnitude constrain"].shape,
                      self.holder_dict["new diffraction magnitude"].shape,
                      self.holder_dict["phase holder"].shape,

                      self.holder_dict["modified support"].shape,
                      self.holder_dict["modified support not"].shape,
                      self.holder_dict["support holder temporary"].shape,

                      self.holder_dict["tmp holder 2"].shape,
                      self.holder_dict["tmp holder 3"].shape,
                      self.holder_dict["tmp holder 4"].shape,
                      ]

        if len(set(shape_list)) != 1:
            print("The data whose shapes that I have checked are :")
            print("self.data_shape, \n" +
                  "self.magnitude.shape, \n" +
                  "self.magnitude_mask.shape, \n" +
                  "self.diffraction.shape, \n" +
                  "self.density.shape, \n" +
                  "self.support.shape, \n" +

                  "self.data_dict[\"magnitude array\"].shape, \n" +
                  "self.data_dict[\"magnitude mask\"].shape, \n" +
                  "self.data_dict[\"support\"].shape, \n" +
                  "self.data_dict[\"density\"].shape, \n" +

                  "self.holder_dict[\"magnitude mask not\"].shape, \n" +
                  "self.holder_dict[\"support not\"].shape, \n" +
                  "self.holder_dict[\"diffraction\"].shape, \n" +
                  "self.holder_dict[\"diffraction with magnitude constrain\"].shape, \n" +
                  "self.holder_dict[\"new diffraction magnitude\"].shape, \n" +
                  "self.holder_dict[\"phase holder\"].shape, \n" +
                  "self.holder_dict[\"modified support\"].shape, \n" +
                  "self.holder_dict[\"modified support not\"].shape, \n" +
                  "self.holder_dict[\"support holder temporary\"].shape, \n" +
                  "self.holder_dict[\"tmp holder 2\"].shape, \n" +
                  "self.holder_dict[\"tmp holder 3\"].shape, \n" +
                  "self.holder_dict[\"tmp holder 4\"].shape,"
                  )
            print("Their corresponding values are :")
            for l in range(len(shape_list)):
                print(shape_list[l])

            print("All the shapes should be the same as the "
                  "{} since this is shape of the magnitude array. ".format(shape_list[0]))

            raise Exception("The data shape are not the same for all the input data, output data"
                            "and intermediate data variables. Please initialize them "
                            "properly.")

    def set_and_check_detector_origin(self, origin=None):
        """
        Because in real experiment, the origin of the diffraction might be difference from the
        center of the image. This function is used to check that.

        :param origin:
        :return:
        """
        if origin:
            self.center_in_pixel = origin

            image_origin = np.divide(self.magnitude.shape, 2)

            # Calculate the distance between the image origin and the new origin
            distance = np.sqrt(np.sum(np.square(image_origin - self.center_in_pixel)))

            if distance >= 2:
                print("The origin of the image and origin of the diffraction differs by"
                      "more than 2 pixels. Do you really want to apply phase retrieval "
                      "algorithms in this case?")
                print("As far as I know, this can add significant artificial effects "
                      "to the result. With high probability, the algorithm will not converge.")

            else:
                print("The distance between the center of the image and the center of "
                      "the diffraction is {:.2f}".format(distance))

        else:
            self.center_in_pixel = np.divide(self.magnitude.shape, 2)
            print("The center of the diffraction is :")
            print(self.center_in_pixel)

    # TODO
    def check_fredel_symmetry(self):
        pass

    ###################################
    # Momentum space
    ###################################
    def fill_detector_gaps(self, gaussian_filter=True, gaussian_sigma=1., bin_num=300):
        """
        Return the magnitude pattern where all the gap pixels are filled with the average value
        of that radial region. A gaussion filter is optional.

        :param gaussian_filter:
        :param gaussian_sigma:
        :param bin_num:
        :return:
        """
        return util.fill_detector_gap(magnitude=self.magnitude,
                                      magnitude_mask=self.magnitude_mask,
                                      origin=self.center_in_pixel,
                                      gaussian_filter=gaussian_filter,
                                      gaussian_sigma=gaussian_sigma,
                                      bin_num=bin_num)

    def set_magnitude(self, magnitude):
        self.magnitude = magnitude

    def set_magnitude_mask(self, magnitude_mask):
        self.magnitude_mask = magnitude_mask

    ###################################
    # Algorithm
    ###################################
    # TODO: Let the user to totally customize the algorithm
    def totally_customize_algorithm(self):
        pass

    def update_holder_dict(self):
        """
        Create variables with proper data shape.
        :return:
        """
        # Create different variables
        self.holder_dict = {"magnitude mask not": np.logical_not(self.magnitude_mask),
                            "support not": np.logical_not(self.support),

                            "diffraction": self.diffraction,
                            "new diffraction magnitude": np.zeros(self.data_shape,
                                                                  dtype=np.complex128),
                            "phase holder": np.zeros(self.data_shape, dtype=np.complex128),
                            "diffraction with magnitude constrain": np.zeros(self.data_shape,
                                                                             dtype=np.complex128),

                            "new density tmp": np.zeros(self.data_shape, dtype=np.float64),

                            "modified support": np.zeros(self.data_shape, dtype=np.bool),
                            "modified support not": np.zeros(self.data_shape, dtype=np.bool),
                            "support holder temporary": np.zeros(self.data_shape, dtype=np.float64),

                            "tmp holder 2": np.zeros(self.data_shape, dtype=np.complex128),
                            "tmp holder 3": np.zeros(self.data_shape, dtype=np.float64),
                            "tmp holder 4": np.zeros(self.data_shape, dtype=np.complex128)
                            }

    def update_input_dict(self):
        """
        Create variables with proper values and data shaps.
        :return:
        """
        self.data_dict = {"magnitude array": self.magnitude,
                          "magnitude mask": self.magnitude_mask,
                          "support": self.support,
                          "density": self.density}

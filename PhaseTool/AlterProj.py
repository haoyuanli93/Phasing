import numpy as np
import copy
from PhaseTool import util

"""
This is the main interface to the applications.

The basic idea is that, the use can create a object to do phase retrieval.
The reason that I would like to use class is that after a short discussion with Zhen,
I realized that class makes it easier for the user to use. Also, in the algorithm, to
accelerate the calculation, I need to initialize a lot of variables before the
calculation. Therefore, it would be very clumsy to use dictionary or the other
methods for the users to use.
"""


class BaseAlterProj:
    def __init__(self, device='cpu'):
        """
        One can use this method to initialize a CDI object and can create a newer one
        with information from this object.

        :param device: Another object of this kind.
        """

        # Meta parameters
        size = 64
        self.dim = 2  # The dimension of the data
        self.data_shape = (size, size)  # The shape of the data in this problem
        self.device = device

        # Algorithm parameters
        self.available_algorithms = ['HIO', 'HPR', 'RAAR', 'GIF-RAAR']
        self.algorithm = "RAAR"
        self.param_dict = {"par_a": 0.,
                           "par_b": 0.,
                           "par_c": 0.,
                           "par_d": 0.,
                           "par_e": 0.,
                           "par_f": 0.,
                           }

        self.iter_num = 200
        self.beta = 0.87 * np.ones(self.iter_num, dtype=np.float64)

        # Whether to save the final diffraction field.
        self.new_diffraction_flag = True

        # Momentum space
        # The position of the center of the diffraction measured in pixel
        self.center_in_pixel = np.array([32., 32.], dtype=np.float64)

        # IO variables
        self.magnitude = np.zeros((size, size), dtype=np.float64)
        self.magnitude_mask = np.ones((size, size), dtype=np.bool)

        self.support = np.ones((size, size), np.bool)
        self.support_not = np.logical_not(self.support)

        self.initial_density = np.ones((size, size))
        self.initial_diffraction = np.ones((size, size), dtype=np.complex128)

        self.new_density = np.zeros((size, size), dtype=np.float64)
        self.new_diffraction = np.zeros((size, size), dtype=np.complex128)

        # Initialize the dictionaries
        """
        Here, the input, output and parameter dictionaries are important because the user might 
        inspect them. However the holder dictionary is not very important since I have created them
        mainly to reduce the memory allocation time. 
        """
        self.input_dict = {"magnitude array": self.magnitude,
                           "magnitude mask": self.magnitude_mask,
                           "magnitude mask not": np.logical_not(self.magnitude_mask),
                           "support": self.support,
                           "support not": self.support_not,
                           "old diffraction": self.initial_diffraction,
                           "old density": self.initial_density,
                           "new diffraction flag": self.new_diffraction_flag
                           }

        self.holder_dict = {str("new diffraction with magnitude" +
                                "constrain"): np.zeros((size, size), dtype=np.complex128),
                            "new diffraction magnitude": np.zeros((size, size), dtype=np.float64),
                            "new density tmp": np.zeros((size, size), dtype=np.float64),
                            "phase holder": np.zeros((size, size), dtype=np.float64),
                            "modified support": np.zeros((size, size), dtype=np.bool),
                            "tmp holder 1": np.zeros((size, size), dtype=np.float64),
                            "tmp holder 2": np.zeros((size, size), dtype=np.complex128),
                            "tmp holder 3": np.zeros((size, size), dtype=np.float64),
                            "tmp holder 4": np.zeros((size, size), dtype=np.complex128)
                            }

        self.output_dict = {"new density": self.new_density,
                            "new diffraction": self.new_diffraction}

        # Shrink wrap behavior
        self.shrink_wrap_on = False

        self.support_fill_holes = False
        self.support_convex_hull = False

        # How many iterations before the threshold decay.
        self.shrink_wrap_decay_rate = 50

        self.shrink_wrap_sigma = 0.
        self.shrink_wrap_sigma_decay_ratio = 0.9

        self.shrink_wrap_threshold_retio = 0.04
        self.shrink_wrap_threshold_decay_ratio = 0.9

        self.shrink_wrap_hist = {'threshold rate history': [],
                                 'sigma history': []}

        self.shrink_wrap_counter = 0

        # Calculation counter
        self.iter_counter = 0

        # History recorder
        self.history = []

    ################################################################################################
    # Simplest initialization
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
        self.magnitude = magnitude
        self.magnitude_mask = magnitude_mask

        # Execute later-on initialization
        if full_initialization:
            self._full_init_from_initez()
        else:
            self._init_from_initez()

    def derive_support_from_autocorrelation(self,
                                            threshold=0.04,
                                            gaussian_filter=True,
                                            sigma=1.0,
                                            fill_detector_gap=False,
                                            bin_num=300):
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
        print("The return of this function if the support array. Please check it to "
              "make sure it makes sense.")

        self.support_not = np.logical_not(self.support)

        # Update the input dictionary
        self.input_dict["support"] = self.support
        self.input_dict["support not"] = self.support_not

        return self.support

    def set_support(self, support):
        """

        :param support:
        :return:
        """
        self.support = support
        self.support_not = np.logical_not(support)

        # Update the input dictionary.
        self.input_dict["support"] = self.support
        self.input_dict["support not"] = self.support_not

    def set_initial_density_and_diffraction(self, density=None, diffraction=None):
        """

        :param density:
        :param diffraction:
        :return:
        """
        if density:
            self.initial_density = density
            print("The initial density is updated.")
        if diffraction:
            self.initial_diffraction = diffraction
            print("The initial diffraction is updated.")

        if density or diffraction:
            self.update_input_dict()

    ################################################################################################
    # Algorithm manipulation and execution
    ################################################################################################
    def set_beta_and_iter_num(self, beta=0.75, iter_num=200, decay=False, decay_rate=7):
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
        else:

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
            self.update_holder_dict()

            # Initialize the parameter dictionary
            if self.algorithm == 'HIO':
                self.param_dict["par_a"] = 1.
                self.param_dict["par_b"] = 0.
                self.param_dict["par_c"] = - 0.87
                self.param_dict["par_d"] = 1.
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

    def update_param_dict_with_beta(self):
        if self.algorithm == "HIO":
            self.param_dict["par_c"] = - self.beta[self.iter_counter]
        elif self.algorithm == "HPR":
            self.param_dict["par_c"] = - self.beta[self.iter_counter]
            self.param_dict["par_e"] = 1 + self.beta[self.iter_counter]
        elif self.algorithm == "HPR":
            self.param_dict["par_c"] = - self.beta[self.iter_counter]
            self.param_dict["par_e"] = 1 + self.beta[self.iter_counter]
        elif self.algorithm == "HPR":
            self.param_dict["par_c"] = - self.beta[self.iter_counter]
            self.param_dict["par_e"] = 1 + self.beta[self.iter_counter]
        else:
            raise Exception("Algorithm {} is not available".format(self.algorithm) +
                            " at present. Please set the " +
                            "self.algorithm propoerty to be one of the following values:\n" +
                            "{} with function".format(self.available_algorithms) +
                            "self.set_algorithm.")

    def execute_algorithm(self,
                          algorithm=None,
                          beta=None,
                          iter_num=None,
                          initial_density=None,
                          shrink_wrap_on=False,
                          ):
        pass

    def shrink_warp_properties(self,
                               threshold_ratio=None,
                               sigma=None,
                               decay_rate=None,
                               threshold_ratio_decay_ratio=0.9,
                               sigma_decay_ratio=0.9,
                               filling_holes=None,
                               convex_hull=None
                               ):
        """
        This is just a temporary implementation of the shrink warp tuning function.
        There are actually, quite a lot of parameters to tune and that has to be consistent with 
        the total iteration number. Therefore, it's complicated. 
        
        So at, present, I will do the folloing things. Instead of getting an array with 
         
        :param threshold_ratio: 
        :param sigma: 
        :param decay_rate: 
        :param threshold_ratio_decay_ratio: 
        :param sigma_decay_ratio: 
        :param filling_holes:
        :param convex_hull:
        :return: 
        """
        if threshold_ratio:
            self.shrink_wrap_threshold_retio = threshold_ratio
            print("The initial threshold ratio of the shrink warp algorithm is set to {}".format(
                threshold_ratio))

        if sigma:
            self.shrink_wrap_sigma = sigma
            print("The initial sigma of the shrink warp algorithm is set to {}".format(sigma))

        if decay_rate:
            self.shrink_wrap_decay_rate = int(decay_rate)
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
            print("The threshold ratio will decay "
                  "to {} * (current ratio)".format(threshold_ratio_decay_ratio) +
                  "after each application of the shrink wrap algorithm. To stop this decaying,"
                  "please set threshold_ratio_decay_ratio=1. when calling this funciton. ")



    def update_shrink_wrap_properties(self):
        pass
        ################################################################################################

    # Details
    ################################################################################################
    ###################################
    # Initialization details
    ###################################
    def _init_from_initez(self):
        """
        This is an internal method used to initialize some more properties when the user has used
        the  method minimal_data_initialize(magnitude, magnitude_mask).

        This method only initializes some not very important variables. The user has to initialize
        the calculation parameters, the initial values and the other stuff.
        :return:
        """

        self.new_density = np.zeros_like(self.magnitude, dtype=np.float64)
        self.new_diffraction = np.zeros_like(self.magnitude, dtype=np.complex128)

        self.input_dict["magnitude array"] = self.magnitude
        self.input_dict["magnitude mask"] = self.magnitude_mask
        self.input_dict["magnitude mask not"] = np.logical_not(self.magnitude_mask)

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

        # Step 1: Finish the most elementary initialization
        self._init_from_initez()

        # Step 2: Initialize the support
        self.derive_support_from_autocorrelation()
        print("Initialize the support array with the auto-correlation array using "
              "default methods with default parameters.")

        # Step 3: Set the initial diffraction and initial density values
        self.set_zeroth_iteration_value(fill_detector_gap=True, phase="Random")

        # Step 4: Initialize the holder variables.
        self.update_input_dict()
        self.update_holder_dict()
        self.update_output_dict()

    def update_holder_dict(self):
        """
        Create variables with proper data shape.
        :return:
        """
        # Create different variables
        self.holder_dict = {str("new diffraction with magnitude" +
                                "constrain"): np.zeros(self.data_shape, dtype=np.complex128),
                            "new diffraction magnitude": np.zeros(self.data_shape,
                                                                  dtype=np.float64),
                            "new density tmp": np.zeros(self.data_shape, dtype=np.float64),
                            "phase holder": np.zeros(self.data_shape, dtype=np.float64),
                            "modified support": np.zeros(self.data_shape, dtype=np.bool),
                            "tmp holder 1": np.zeros(self.data_shape, dtype=np.float64),
                            "tmp holder 2": np.zeros(self.data_shape, dtype=np.complex128),
                            "tmp holder 3": np.zeros(self.data_shape, dtype=np.float64),
                            "tmp holder 4": np.zeros(self.data_shape, dtype=np.complex128)
                            }

    def update_input_dict(self):
        """
        Create variables with proper values and data shaps.
        :return:
        """
        self.input_dict = {"magnitude array": self.magnitude,
                           "magnitude mask": self.magnitude_mask,
                           "magnitude mask not": np.logical_not(self.magnitude_mask),
                           "support": self.support,
                           "old diffraction": self.initial_diffraction,
                           "old density": self.initial_density,
                           "new diffraction flag": self.new_diffraction_flag
                           }

    def update_output_dict(self):
        """

        :return:
        """
        self.output_dict = {"new density": self.new_density,
                            "new diffraction": self.new_diffraction}

    def set_zeroth_iteration_value(self, fill_detector_gap=False, phase="Random"):

        # Step 1: Get the phase
        if phase in ('Random', 'Zero', 'Minimal'):
            if phase == "Random":

                # Create a central symmetric phase array
                tmp1 = np.random.rand(*self.magnitude.shape)
                tmp2 = np.copy(tmp1)
                for l in range(self.dim):
                    tmp2 = np.flip(m=tmp2, axis=l)

                phase_array = np.exp(1j * np.pi * (tmp1 - tmp2))

            elif phase == "Zero":
                phase_array = np.ones_like(self.magnitude)
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
        self.initial_diffraction = np.multiply(phase_array, magnitude_tmp)
        self.initial_density = np.fft.ifftn(self.initial_diffraction).real

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
                      self.initial_diffraction.shape,
                      self.initial_density.shape,
                      self.support.shape,
                      self.new_density.shape,
                      self.new_diffraction.shape,
                      self.input_dict["magnitude array"].shape,
                      self.input_dict["magnitude mask"].shape,
                      self.input_dict["magnitude mask not"].shape,
                      self.input_dict["support"].shape,
                      self.input_dict["old diffraction"].shape,
                      self.input_dict["old density"].shape,
                      self.holder_dict["new diffraction with magnitude constrain"].shape,
                      self.holder_dict["new diffraction magnitude"].shape,
                      self.holder_dict["phase holder"].shape,
                      self.holder_dict["modified support"].shape,
                      self.holder_dict["tmp holder 1"].shape,
                      self.holder_dict["tmp holder 2"].shape,
                      self.holder_dict["tmp holder 3"].shape,
                      self.holder_dict["tmp holder 4"].shape,
                      self.output_dict["new diffraction"].shape,
                      self.output_dict["new density"].shape
                      ]

        if len(set(shape_list)) != 1:
            print("The data whose shapes that I have checked are :")
            print("self.data_shape,\n" +
                  "self.magnitude.shape,\n" +
                  "self.magnitude_mask.shape\n" +
                  "self.initial_diffraction.shape\n" +
                  "self.initial_density.shape\n" +
                  "self.support.shape\n" +
                  "self.new_density.shape\n" +
                  "self.new_diffraction.shape\n" +
                  "self.input_dict[\"magnitude array\"].shape\n" +
                  "self.input_dict[\"magnitude mask\"].shape\n" +
                  "self.input_dict[\"magnitude mask not\"].shape\n" +
                  "self.input_dict[\"support\"].shape\n" +
                  "self.input_dict[\"old diffraction\"].shape\n" +
                  "self.input_dict[\"old density\"].shape\n" +
                  "self.holder_dict[\"new diffraction with magnitude constrain\"].shape\n" +
                  "self.holder_dict[\"new diffraction magnitude\"].shape\n" +
                  "self.holder_dict[\"phase holder\"].shape\n" +
                  "self.holder_dict[\"modified support\"].shape\n" +
                  "self.holder_dict[\"tmp holder 1\"].shape\n" +
                  "self.holder_dict[\"tmp holder 2\"].shape\n" +
                  "self.holder_dict[\"tmp holder 3\"].shape\n" +
                  "self.holder_dict[\"tmp holder 4\"].shape\n" +
                  "self.output_dict[\"new diffraction\"].shape\n" +
                  "self.output_dict[\"new density\"].shape\n"
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
    # Set beta and iteration number
    ###################################

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
    def totally_customize_algorithm(self):
        pass

    def get_device(self):
        """

        :return:
        """
        return self.device

    def set_device(self, device):
        """

        :param device:
        :return:
        """
        if device in ('cpu', 'gpu'):
            self.device = device
        else:
            raise Exception("The device can only be 'cpu' or 'gpu'.")

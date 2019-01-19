import numpy as np
import copy

"""
This is the main interface to the applications.

The basic idea is that, the use can create a object to do phase retrieval.
The reason that I would like to use class is that after a short discussion with Zhen,
I realized that class makes it easier for the user to use. Also, in the algorithm, to
accelerate the calculation, I need to initialize a lot of variables before the
calculation. Therefore, it would be very clumsy to use dictionary or the other
methods for the users to use.
"""


class BaseCDI:
    def __init__(self, device='cpu', source=None):
        """
        One can use this method to initialize a CDI object and can create a newer one
        with information from this object.

        :param source: Another object of this kind.
        """

        # Meta parameters
        size = 64
        self.dim = 2  # The dimension of the data
        self.data_shape = (size, size)  # The shape of the data in this problem
        self.device = device

        """
        There are situations where the users would like to see the logs directly while they 
        want to save them to the disk. This parameter tune that feature. The log is always saved 
        to the disk in a log file. 
        
        If self.verbose = False, not information is shown in the standard iostream.
        If self.verbose = True, information will be shown on the screen directly.
        """
        self.verbose = True
        self.log_address = "default"

        # Algorithm parameters
        self.algorithm = "RAAR"
        self.param_dict = {"par_a": 0,
                           "par_b": 0,
                           "par_c": 0,
                           "par_d": 0,
                           "par_e": 0,
                           "par_f": 0,
                           }

        self.iter_num = 200
        self.beta = 0.87 * np.ones(self.iter_num, dtype=np.float64)

        # Momentum space

        self.magnitude = np.zeros((size, size), dtype=np.float64)
        self.magnitude_mask = np.ones((size, size), dtype=np.bool)

        # This flag is self-evident.
        self.momentum_space_initialization_flag = False
        self.initial_diffraction = np.ones((size, size), dtype=np.complex128)

        # Real space
        self.support = np.ones((size, size), np.bool)
        self.initial_density = np.ones((size, size))

        # Output flag
        # Whether to output the new diffraction pattern
        self.new_diffraction_flag = False

        # Output arrays
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

        if source:
            if source.__class__.__name__ == "BaseCDI":
                self.deep_copy(source=source)

    def minimal_data_initialize(self, magnitude, magnitude_mask, full_initialization=True):
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
            self._full_init_from_mdinit()
        else:
            self._init_from_mdinit()

    def _init_from_mdinit(self):
        """
        This is an internal method used to initialize some more properties when the user has used
        the  method minimal_data_initialize(magnitude, magnitude_mask).

        This method only initializes some not very important variables. The user has to initialize
        the calculation parameters, the initial values and the other stuff.
        :return:
        """

        self.input_dict["magnitude array"] = self.magnitude
        self.input_dict["magnitude mask"] = self.magnitude_mask
        self.input_dict["magnitude mask not"] = np.logical_not(self.magnitude_mask)

        self.dim = len(self.magnitude.shape)
        self.data_shape = copy.deepcopy(self.magnitude.shape)

    def check_self_consistency(self):
        """
        This function check if all the properties in this object is consistent with each other.
        :return:
        """
        pass

    def set_beta(self, beta):
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
        :return:
        """
        self.beta = beta

        if isinstance(beta, (list, tuple, np.ndarray)):
            self.beta = np.array(beta)
            self.iter_num = self.beta.shape[0]
        else:
            self.beta = beta * np.ones(self.iter_num, dtype=np.float64)

    def use_default_beta(self, iter_num=200, decaying=True):
        pass

    def _full_init_from_mdinit(self):
        """
        This is an internal method used to initialize some more properties when the user has
        used the method minimal_data_initialize(magnitude, magnitude_mask).

        In short, this function initialize all

        :return:
        """
        autocorrelation = np.fft.ifftn(np.square(self.magnitude)).real
        

    def initialize_data(self, support):
        pass

    def totally_customize_algorithm(self):
        pass

    def start_with_random_phase(self):
        pass

    def deep_copy(self, source):
        pass

    def set_support(self, support):
        self.support = support

    def set_magnitude(self, magnitude):
        self.magnitude = magnitude

    def set_magnitude_mask(self, magnitude_mask):
        self.magnitude_mask = magnitude_mask

    def set_algorithm(self, alg_name):
        self.algorithm = alg_name

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

    def execute_algorithm(self):
        pass

    def update_param_dict_with_beta(self):
        pass

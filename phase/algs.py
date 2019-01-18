import numpy as np

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
        self.shape = (size, size)  # The shape of the data in this problem
        self.device = device

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

        # Real space
        self.support = np.ones((size, size), np.bool)
        self.initial_density = np.ones((size, size))

        self.holder_dict = {}
        self.input_dict = {}
        self.param_dict = {}
        self.output_dict = {}

        if source:
            if source.__class__.__name__ == "BaseCDI":
                self.deep_copy(source=source)

    def initialize_data(self):
        pass

    def totally_customize_algorithm(self):
        pass

    def start_with_random_phase(self):
        pass

    def deep_copy(self, source):
        pass

    def set_support(self):
        pass

    def set_magnitude(self):
        pass

    def set_magnitude_mask(self):
        pass

    def set_algorithm(self):
        pass

    def get_device(self):
        pass

    def set_device(self):
        pass

    def execute_algorithm(self):
        pass

    def update_param_dict_with_beta(self):
        pass

"""
Usually, it is not enough to use a single algorithm to recover the phase.
This script put several algorithms together and implements some common
practice to make it easier for the users to use.
"""
import copy


class AlterProjChain:
    """
    This is an auxiliary class. Even though it can be, I have intentionally not to make it to
    be a super class of the CpuAlterProj and the GpuAlterProj object for simplicity.

    It just sounds too complicated to me and I am not sure if i am able to do it.
    Therefore, I guess this code is much longer than it has to be.
    """

    def __init__(self, intensity=None, detector_mask=None, keep_full_history=False, device='cpu'):
        # Meta data to guide the calculation
        self.device = device
        self.dim = 2  # This property is not useful for cpu calculation but is useful for gpu.
        self.algorithm_sequence = copy.deepcopy(default_alter_proj_chain_1)

        # Necessary input data
        self.intensity = None
        self.detector_mask = None

        if intensity:
            self.intensity = intensity

        if detector_mask:
            self.detector_mask = detector_mask

        # Important output
        self.support = None
        self.support_history = []
        self.density_history = []
        self.magnitude_history = []

        """
        This contains all the intermediate AlterProj objects created for the calculation.  
        """
        self.keep_full_history = keep_full_history
        # self.keep_full_history:   Whether to update the alter_proj_obj_history list
        self.alter_proj_obj_history = []

    def initialize_data(self, intensity, detector_mask):
        """
        This function only initialize the
        :return:
        """
        self.intensity = intensity
        self.detector_mask = detector_mask

        self.dim = len(self.intensity.shape)

    def _create_algorithm_object(self, algorithm_info):
        """
        Create an algorithm object according to the info in the algorithm info dictionary
        :param algorithm_info: 
        :return: 
        """
        pass

    def _modify_algorithm_object(self, algorithm_info):
        """
        After the first stage, one will copy from the existing 
        :param algorithm_info: 
        :return: 
        """

    def execute_algorithm_sequence(self):

        # Counter for algorithm dictionaries
        for ctr in range(len(self.algorithm_sequence)):
            if ctr == 0:
                alter_proj = self._create_algorithm_object(self.algorithm_sequence[ctr])

                # TODO Finish the calculation
                pass
                if self.keep_full_history:
                    self.alter_proj_obj_history.append(alter_proj)
            else:
                pass

    def set_algorithm_sequence(self):
        pass

    def use_default_algorithm_sequence(self, idx):
        pass


"""
This chain assumes that there is not detector gap. Therefore, the auto-correlation is a good start 
point for the reconstruction   
"""
default_alter_proj_chain_1 = [

    ###############################################################################################
    # 1st Dict: It initializes the object, gets support, set shrink warp, do some calculation.
    ###############################################################################################
    {

        # Set the algorithm properties
        'AlgName': 'RAAR',  # Algorithm name
        'IterNum': 1200,  # Iternation number
        'InitBeta': 0.87,  # The initial beta value
        'BetaDecay': False,  # Whether the beta value will decay after several iterations
        'BetaDecayRate': None,
        # Since BetaDecay is disabled, there is not need for this value.
        # Usually, this value is set to 30.

        # Set the initial support properties
        'InitSupport Type': 'Auto-correlation',  # Initial support type
        'InitSupport': None,
        # Because one derive the initial support
        # from auto-correlation, this is not used

        # Set the shrink-wrap properties
        'ShrinkWrap Flag': False  # Whether to use ShrinkWrap algorithm to update the support
    },
    {},
    {}]

"""
# Step 1: Create a object
alter_proj = PhaseTool.AlterProj.BaseAlterProj()

# Step 2: Initialize the object with the data
alter_proj.initialize_easy(magnitude=np.fft.ifftshift(magnitude),
                           magnitude_mask=np.ones_like(magnitude, dtype=np.bool),
                           full_initialization=False)
# Step 3: Set initial guess
alter_proj.set_support(support=alter_proj.derive_support_from_autocorrelation(threshold=0.04,
                                                                              gaussian_filter=True,
                                                                              sigma=1.0,
                                                                              fill_detector_gap=False,
                                                                              bin_num=300))

alter_proj.set_zeroth_iteration_value(fill_detector_gap=False, phase="Random")

alter_proj.update_input_dict()

alter_proj.shrink_warp_properties(on=False,
                                  threshold_ratio=0.04,
                                  sigma=10.,
                                  decay_rate=50,
                                  threshold_ratio_decay_ratio=1.0,
                                  sigma_decay_ratio=0.99,
                                  filling_holes=False,
                                  convex_hull=False)

alter_proj.set_beta_and_iter_num(beta=0.75,
                                 iter_num=1200,
                                 decay=True,
                                 decay_rate=20)

alter_proj.set_algorithm(alg_name="RAAR")
"""

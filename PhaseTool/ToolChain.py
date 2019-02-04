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
        self.alter_proj_obj = None  # The reference to find the AlterProj object
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

    def show_algorithm_sequence(self):
        pass

    def use_default_algorithm_sequence(self, idx):
        pass


###############################################################################################
#
#   Below are some default chain parameters
#
###############################################################################################

"""
This chain assumes that there is not detector gap. Therefore, the auto-correlation is a good start 
point for the reconstruction   
"""
default_alter_proj_chain_1 = [

    ###############################################################################################
    # 1st Dict: It initializes the object, gets support, set shrink warp, do some calculation.
    ###############################################################################################
    {
        # Group 1: Set the algorithm properties
        'AlgName': 'RAAR',  # Algorithm name
        'IterNum': 1200,  # Iternation number
        'InitBeta': 0.87,  # The initial beta value
        'BetaDecay': True,  # Whether the beta value will decay after several iterations
        'BetaDecayRate': 20,  # How the beta value decays

        # Group 2: Set the initial support properties
        'InitSupport Type': 'Auto-correlation',  # Initial support type
        # The following entry is for Type 'Assigned'. Because one derive the initial support
        # from auto-correlation, this is not used in this step in this chain.
        'InitSupport': None,
        'InitSupport Threshold': 0.04,  # The threshold to get the support
        'InitSupport Gaussian Filter': True,  # Whether to use Gaussian filter to get the support
        'InitSupport Gaussian sigma': 1.0,
        # Whether to fill the detector gaps when calculating the auto-correlation
        'InitSupport Fill Detector Gaps': True,

        # Group 3: Set the initial density properties
        'InitDensity Type': "Support",  # Derive the density from the support.
        'InitDensity Phase': "Random",

        # Group 4: Set the shrink-wrap properties
        'ShrinkWrap Flag': False,  # Whether to use ShrinkWrap algorithm to update the support
        'ShrinkWrap Threshold Ratio': 0.04,  # This is the default value. Not using this entry.
        'ShrinkWrap Sigma': 10.,  # Default value
        'ShrinkWrap Decay Rate': 50,  # Default Value
        'ShrinkWrap Threshold Decay Ratio': 1.0,  # Default value.
        'ShrinkWrap Sigma Decay Ratio': 0.99,  # Default value.for
        'ShrinkWrap Filling Holes': True,
        'ShrinkWrap ConvexHull': False,
    },

    ###############################################################################################
    # 2nd Dict: Copy the previous reconstruction object. Notice that I am writing it in a simpler
    # way since I do not change much parameters. However, you can just specify all the parameters
    # just like the first dictionary.
    ###############################################################################################
    {
        # Group 1: Set the algorithm properties
        """
        Since I do not want to change any parameters about the algorithm itself, 
        I do not need to specify them again. If one wants to change any parameters here, one can
        copy the corresponding entries from the previous dictionary and make the corresponding 
        modifications. 
        """

        # Group 2: Set the initial support properties
        'InitSupport Type': 'Assigned',  # Initial support type
        # The following entry is for Type 'Assigned'.
        'InitSupport': 'Current Support',
        # This means does not change anything since this second object is a deepcopy of the
        # previous object

        # Group 3: Set the initial density properties
        'InitDensity Type': "Assigned",  # Use the result from previous object

        # Group 4: Set the shrink-wrap properties
        'ShrinkWrap Flag': False,  # Whether to use ShrinkWrap algorithm to update the support
        'ShrinkWrap Threshold Ratio': 0.04,  # This is the default value. Not using this entry.
        'ShrinkWrap Sigma': 10.,  # Default value
        'ShrinkWrap Decay Rate': 50,  # Default Value
        'ShrinkWrap Threshold Decay Ratio': 1.0,  # Default value.
        'ShrinkWrap Sigma Decay Ratio': 0.99,  # Default value.for
        'ShrinkWrap Filling Holes': True,
        'ShrinkWrap ConvexHull': False,
    },

    ###############################################################################################
    # 3rd Dict: It initializes the object, gets support, set shrink warp, do some calculation.
    ###############################################################################################
    {
        # Group 1: Set the algorithm properties
        'AlgName': 'RAAR',  # Algorithm name
        'IterNum': 1200,  # Iternation number
        'InitBeta': 0.87,  # The initial beta value
        'BetaDecay': True,  # Whether the beta value will decay after several iterations
        'BetaDecayRate': 20,  # How the beta value decays

        # Group 2: Set the initial support properties
        'InitSupport Type': 'Auto-correlation',  # Initial support type
        # The following entry is for Type 'Assigned'. Because one derive the initial support
        # from auto-correlation, this is not used in this step in this chain.
        'InitSupport': None,
        'InitSupport Threshold': 0.04,  # The threshold to get the support
        'InitSupport Gaussian Filter': True,  # Whether to use Gaussian filter to get the support
        'InitSupport Gaussian sigma': 1.0,
        # Whether to fill the detector gaps when calculating the auto-correlation
        'InitSupport Fill Detector Gaps': True,

        # Group 3: Set the initial density properties
        'InitDensity Type': "Support",  # Derive the density from the support.
        'InitDensity Phase': "Random",

        # Group 4: Set the shrink-wrap properties
        'ShrinkWrap Flag': False,  # Whether to use ShrinkWrap algorithm to update the support
        'ShrinkWrap Threshold Ratio': 0.04,  # This is the default value. Not using this entry.
        'ShrinkWrap Sigma': 10.,  # Default value
        'ShrinkWrap Decay Rate': 50,  # Default Value
        'ShrinkWrap Threshold Decay Ratio': 1.0,  # Default value.
        'ShrinkWrap Sigma Decay Ratio': 0.99,  # Default value.for
        'ShrinkWrap Filling Holes': True,
        'ShrinkWrap ConvexHull': False,
    }
]

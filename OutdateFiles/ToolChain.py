"""
Usually, it is not enough to use a single algorithm to recover the phase.
This script put several algorithms together and implements some common
practice to make it easier for the users to use.
"""
import copy
from PhaseTool.AlterProj import CpuAlterProj
import numpy as np


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

        if not (intensity is None):
            self.intensity = intensity

        if not (detector_mask is None):
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

    def _create_algorithm_object(self, alg_info, device=None):
        """
        Create an algorithm object according to the info in the algorithm info dictionary
        :param alg_info:
        :param device: Later, when we have the GPU version, this argument will be used.
        :return: 
        """

        # TODO: Distinguish different device configurations
        if device:
            pass

        # Step 1: Create the object
        self.alter_proj_obj = CpuAlterProj()

        # Step 2: Initialize the object
        self.alter_proj_obj.initialize_easy(magnitude=np.fft.fftshift(np.sqrt(self.intensity)),
                                            magnitude_mask=np.fft.fftshift(self.detector_mask),
                                            full_initialization=False)

        # Step 3: Get the support
        if alg_info['InitSupport Type'] == 'Auto-correlation':

            # Test
            print("Initialize with auto-correlation information.")

            # Derive the support from the auto-correlation
            _ = self.alter_proj_obj.use_auto_support(
                threshold=alg_info['InitSupport Threshold'],
                gaussian_filter=alg_info['InitSupport Gaussian Filter'],
                sigma=alg_info['InitSupport Gaussian sigma'],
                fill_detector_gap=alg_info['InitSupport Fill Detector Gaps'])

        # Deal with the case where the user want to use their own support
        elif alg_info['InitSupport Type'] == 'Assigned':

            # Use previous support, just do not change anything
            if alg_info['InitSupport'] == 'Current Support':
                # This means does not change anything
                pass

            # Use a specified numpy array compatible with the intensity array
            elif type(alg_info['InitSupport']).__name__ == 'ndarray':

                # Check type and shape before assigning the value.
                if alg_info['InitSupport'].shape == self.intensity.shape:
                    self.alter_proj_obj.set_support(support=alg_info['InitSupport'])
                else:
                    raise Exception("The shape of the assigned initial support has to be the same "
                                    "as that of the intensity array.")

            else:  # Deal with unexpected values
                raise Exception("Sorry, at this time, the logic of this program is not very "
                                "complete, please set the entry 'InitSupport' to "
                                "'Current Support' use the existing support or a compatible "
                                "numpy array since they are the only supported options. ")

        else:  # Deal with unexpected values
            raise Exception("Sorry, at this time, the logic of this program is not very "
                            "complete, please set the entry 'InitSupport Type' to "
                            "'Auto-correlation' if you want to derive the support from the "
                            "auto-correlation or 'Assigned' if you want to use existing value "
                            "or assign a new value. ")

        # Step 4: Set the initial density value
        # Case 1: The user want to derive the density
        if alg_info['InitDensity Type'] == 'Derived':
            self.alter_proj_obj.derive_initial_density(
                fill_detector_gap=alg_info['InitDensity Fill Detector Gaps'],
                method=alg_info['InitDensity Deriving Method'])

        # Case 2: The user want to assign the initial density value
        elif alg_info['InitDensity Type'] == 'Assigned':

            # If use the previous result, does not change anything
            if alg_info['InitDensity'] == 'Current Density':
                pass

            # Use a specified numpy array compatible with the intensity array
            elif type(alg_info['InitDensity']).__name__ == 'ndarray':

                # Check type and shape before assigning the value.

                if alg_info['InitDensity'].shape == self.intensity.shape:
                    self.alter_proj_obj.set_initial_density(density=alg_info['InitDensity'])
                else:
                    raise Exception("The shape of the assigned initial density has to be the same "
                                    "as that of the intensity array.")

            else:  # Deal with unexpected values
                raise Exception("Sorry, at this time, the logic of this program is not very "
                                "complete, please set the entry 'InitDensity' to "
                                "'Current Density' use the existing density or a compatible "
                                "numpy array since they are the only supported options. ")

        else:  # Deal with unexpected values
            raise Exception("Sorry, at this time, the logic of this program is not very "
                            "complete, please set the entry 'InitDensity Type' to "
                            "'Auto' if you want to derive the density "
                            "auto-correlation or 'Assigned' if you want to use exising value "
                            "or assign a new value. ")

        # Step 5: Specify the ShrinkWrap parameters
        if alg_info['ShrinkWrap Flag']:
            self.alter_proj_obj.shrink_warp_properties(
                on=alg_info['ShrinkWrap Flag'],
                threshold_ratio=alg_info['ShrinkWrap Threshold Ratio'],
                sigma=alg_info['ShrinkWrap Sigma'],
                decay_rate=alg_info['ShrinkWrap Decay Rate'],
                threshold_ratio_decay_ratio=alg_info['ShrinkWrap Threshold Decay Ratio'],
                sigma_decay_ratio=alg_info['ShrinkWrap Sigma Decay Ratio'],
                filling_holes=alg_info['ShrinkWrap Filling Holes'],
                convex_hull=alg_info['ShrinkWrap ConvexHull'])

        else:
            # Turn off the shrink wrap process
            self.alter_proj_obj.shrink_warp_properties(on=False)

        # Step 6: Set the Alternating Projection Algorithm parameters.
        if 'AlgName' in alg_info:
            self.alter_proj_obj.set_algorithm(alg_name=alg_info['AlgName'])

            if alg_info['AlgName'] == 'ER':
                self.alter_proj_obj.set_beta_and_iter_num(beta=None,
                                                          iter_num=alg_info['IterNum'])

            else:
                self.alter_proj_obj.set_beta_and_iter_num(beta=alg_info['InitBeta'],
                                                          iter_num=alg_info['IterNum'],
                                                          decay=alg_info['BetaDecay'],
                                                          decay_rate=alg_info['BetaDecayRate'])

        # Step 7: Update the input dict and the holder dict
        self.alter_proj_obj.update_input_dict()
        self.alter_proj_obj.update_holder_dict()

    def _modify_algorithm_object(self, alg_info):
        """
        After the first stage, one will copy from the existing 
        :param alg_info:
        :return: 
        """

        # Step 1: Modify the support
        if alg_info['InitSupport Type'] == 'Auto-correlation':
            # Derive the support from the auto-correlation
            _ = self.alter_proj_obj.use_auto_support(
                threshold=alg_info['InitSupport Threshold'],
                gaussian_filter=alg_info['InitSupport Gaussian Filter'],
                sigma=alg_info['InitSupport Gaussian sigma'],
                fill_detector_gap=alg_info['InitSupport Fill Detector Gaps'],
                bin_num=alg_info['InitSupport Bin Num'])

        # Deal with the case where the user want to use their own support
        elif alg_info['InitSupport Type'] == 'Assigned':

            # Use previous support, just do not change anything
            if alg_info['InitSupport'] == 'Current Support':
                # This means does not change anything
                pass

            # Use a specified numpy array compatible with the intensity array
            elif type(alg_info['InitSupport']).__name__ == 'ndarray':

                # Check type and shape before assigning the value.
                if alg_info['InitSupport'].shape == self.intensity.shape:
                    self.alter_proj_obj.set_support(support=alg_info['InitSupport'])
                else:
                    raise Exception("The shape of the assigned initial support has to be the same "
                                    "as that of the intensity array.")

            else:  # Deal with unexpected values
                raise Exception("Sorry, at this time, the logic of this program is not very "
                                "complete, please set the entry 'InitSupport' to "
                                "'Current Support' use the existing support or a compatible "
                                "numpy array since they are the only supported options. ")

        else:  # Deal with unexpected values
            raise Exception("Sorry, at this time, the logic of this program is not very "
                            "complete, please set the entry 'InitSupport Type' to "
                            "'Auto-correlation' if you want to derive the support from the "
                            "auto-correlation or 'Assigned' if you want to use existing value "
                            "or assign a new value. ")

        # Step 2: Set the initial density value
        # Case 1: The user want to derive the density
        if alg_info['InitDensity Type'] == 'Derived':
            self.alter_proj_obj.derive_initial_density(
                fill_detector_gap=alg_info['InitDensity Fill Detector Gaps'],
                method=alg_info['InitDensity Deriving Method'])

        # Case 2: The user want to assign the initial density value
        elif alg_info['InitDensity Type'] == 'Assigned':

            # If use the previous result, does not change anything
            if alg_info['InitDensity'] == 'Current Density':
                pass

            # Use a specified numpy array compatible with the intensity array
            elif type(alg_info['InitDensity']).__name__ == 'ndarray':

                # Check type and shape before assigning the value.

                if alg_info['InitDensity'].shape == self.intensity.shape:
                    self.alter_proj_obj.set_initial_density(density=alg_info['InitDensity'])
                else:
                    raise Exception("The shape of the assigned initial density has to be the same "
                                    "as that of the intensity array.")

            else:  # Deal with unexpected values
                raise Exception("Sorry, at this time, the logic of this program is not very "
                                "complete, please set the entry 'InitDensity' to "
                                "'Current Density' use the existing density or a compatible "
                                "numpy array since they are the only supported options. ")

        else:  # Deal with unexpected values
            raise Exception("Sorry, at this time, the logic of this program is not very "
                            "complete, please set the entry 'InitDensity Type' to "
                            "'Auto' if you want to derive the density "
                            "auto-correlation or 'Assigned' if you want to use exising value "
                            "or assign a new value. ")

        # Step 5: Specify the ShrinkWrap parameters
        if alg_info['ShrinkWrap Flag']:
            self.alter_proj_obj.shrink_warp_properties(
                on=alg_info['ShrinkWrap Flag'],
                threshold_ratio=alg_info['ShrinkWrap Threshold Ratio'],
                sigma=alg_info['ShrinkWrap Sigma'],
                decay_rate=alg_info['ShrinkWrap Decay Rate'],
                threshold_ratio_decay_ratio=alg_info['ShrinkWrap Threshold Decay Ratio'],
                sigma_decay_ratio=alg_info['ShrinkWrap Sigma Decay Ratio'],
                filling_holes=alg_info['ShrinkWrap Filling Holes'],
                convex_hull=alg_info['ShrinkWrap ConvexHull'])

        else:
            # Turn off the shrink wrap process
            self.alter_proj_obj.shrink_warp_properties(on=False)

        # Step 6: Set the Alternating Projection Algorithm parameters.
        if 'AlgName' in alg_info:
            self.alter_proj_obj.set_algorithm(alg_name=alg_info['AlgName'])

            if alg_info['AlgName'] == 'ER':
                self.alter_proj_obj.set_beta_and_iter_num(iter_num=alg_info['IterNum'])

            else:
                self.alter_proj_obj.set_beta_and_iter_num(beta=alg_info['InitBeta'],
                                                          iter_num=alg_info['IterNum'],
                                                          decay=alg_info['BetaDecay'],
                                                          decay_rate=alg_info['BetaDecayRate'])

        # Step 7: Update the input dict and the holder dict
        self.alter_proj_obj.update_input_dict()
        self.alter_proj_obj.update_holder_dict()

    def execute_algorithm_sequence(self):
        """
        Execute the algorithm sequence.

        :return: None
        """

        print("")
        print("***********************************************************************************")
        print("******************     Start The Calculation      *********************************")
        print("***********************************************************************************")
        print("")

        # Counter for algorithm dictionaries
        for ctr in range(len(self.algorithm_sequence)):

            print("")
            print("***------------------------------------------------------------------------***")
            print("Begin the No.{} algorithm in this sequence.".format(ctr))

            if ctr == 0:

                # During the first iteration, create the project
                self._create_algorithm_object(self.algorithm_sequence[ctr])

                # Execute the algorithm
                self.alter_proj_obj.execute_algorithm()

                if self.keep_full_history:
                    self.alter_proj_obj_history.append(copy.deepcopy(self.alter_proj_obj))
            else:
                # During the first iteration, create the project
                self._modify_algorithm_object(self.algorithm_sequence[ctr])

                # Execute the algorithm
                self.alter_proj_obj.execute_algorithm()

                if self.keep_full_history:
                    self.alter_proj_obj_history.append(copy.deepcopy(self.alter_proj_obj))

            print("***------------------------------------------------------------------------***")
            print("")

        print("Finished all the steps in the calculation.")

    def set_algorithm_sequence(self, alg_sequence):
        """
        Use customized algorithm sequence.

        :param alg_sequence:
        :return:
        """
        if type(alg_sequence).__name__ == 'list':
            self.algorithm_sequence = alg_sequence
        else:
            raise Exception("The alg_sequence has to be a list, even if you only use only one "
                            "algorithm and has specified only one dictionary.")

    def show_introduction_and_algorithm_sequence(self, show_detail=False):
        """
        This function shows the calculation sequence in a understandable way.

        :return:
        """

        # Step 0: Give a fixed introduction
        print("----------------------------Part 0: Introduction-----------------------------------")
        print("When doing phase retrieval, it seems a little bit difficult to do it with a single "
              "algorithm. Therefore I created this AlterProjChain object to organize a sequence of "
              "different projection methods. Therefore, this object is not fundamental. You can "
              "definitely combine CpuAlterProj objects to finished whatever you can do with this "
              "object and more.")
        print("")
        print("In this object, the calculation sequence is controlled with a list called "
              "algorithm_sequence. Each element in this list is a dictionary. Each dictionary "
              "contains complete information for one group of alternating projection algorithm.")
        print("")

        # Step 1: Show briefly summarize the this calculation sequence.
        print("----------------------------Part 1: Current Sequence Summary-----------------------")

        alg_num = len(self.algorithm_sequence)
        # Deal with the case when the user has not initialized this sequence.
        if alg_num == 0:
            print("There is no algorithm in the algorithm sequence. It seems that you have not "
                  "initialized this algorithm sequence. Please initialize the sequence with "
                  "function set_algorithm_sequence or use_default_algorithm_sequence.")

        else:
            print("At present there are totally {} elements in the algorithm sequence.".format(
                alg_num))
            print("They are respectively:")
            for l in range(alg_num):

                print("")
                print("For algorithm No.{}".format(l))

                alg_info = self.algorithm_sequence[l]
                print("Algorithm Name:{}".format(alg_info['AlgName']))
                print("Iteration Number:{}".format(alg_info['IterNum']))
                print("Use Shrink Wrap:{}".format(alg_info['ShrinkWrap Flag']))
                print("")

        # Step 2: Check if one needs to show the details
        if show_detail:
            print("----------------------------Part 2: "
                  "Detail---------------------------------------")
            print("Since you would like to see the detail of the sequence, the elements in the "
                  "algorithm sequence is printed in sequence. The detailed explanation of each "
                  "entry of the dictionary can be found on the github and in the source code.")
            for l in range(alg_num):
                print("For algorithm No.{}".format(l))
                print(self.algorithm_sequence[l])

        print("----------------------------End of the introduction--------------------------------")
        print('')
        print('')

    def use_default_algorithm_sequence(self, idx):
        """
        Switch between different algorithm chains

        :param idx:
        :return:
        """
        if idx == 1:
            self.algorithm_sequence = default_alter_proj_chain_1
        else:
            raise Exception("Sorry, at present, idx has to be 1 since I have only prepared "
                            "1 default chain.")

    def get_default_algorithm_sequence(self, idx):
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
        'InitSupport Fill Detector Gaps': False,

        # Group 3: Set the initial density properties
        'InitDensity Type': "Derived",  # Derive the density from the support.
        'InitDensity': None,  # Since it's derived, no need for the input
        'InitDensity Deriving Method': "Random",
        'InitDensity Fill Detector Gaps': True,

        # Group 4: Set the shrink-wrap properties
        'ShrinkWrap Flag': False,  # Whether to use ShrinkWrap algorithm to update the support
        'ShrinkWrap Threshold Ratio': 0.04,  # This is the default value. Not using this entry.
        'ShrinkWrap Sigma': 5.,  # Default value
        'ShrinkWrap Decay Rate': 30,  # Default Value
        'ShrinkWrap Threshold Decay Ratio': 1.0,  # Default value.
        'ShrinkWrap Sigma Decay Ratio': 0.95,  # Default value.for
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
        'AlgName': 'RAAR',  # Algorithm name
        'IterNum': 1200,  # Iternation number
        'InitBeta': 0.87,  # The initial beta value
        'BetaDecay': True,  # Whether the beta value will decay after several iterations
        'BetaDecayRate': 20,  # How the beta value decays

        # Group 2: Set the initial support properties
        'InitSupport Type': 'Assigned',  # Initial support type
        # The following entry is for Type 'Assigned'.
        'InitSupport': 'Current Support',
        # This means does not change anything since this second object is a deepcopy of the
        # previous object

        # Group 3: Set the initial density properties
        'InitDensity Type': "Assigned",  # Use the result from previous object
        'InitDensity': 'Current Density',
        # For the other parameters, appeared in the first dict, since they are not used or not
        # modified, I do not need to specify them again. If one wants to change any parameters
        # here, one can copy the corresponding entries from the previous dictionary and make the
        # corresponding modifications.

        # Group 4: Set the shrink-wrap properties
        'ShrinkWrap Flag': True,  # Whether to use ShrinkWrap algorithm to update the support
        'ShrinkWrap Threshold Ratio': 0.04,  # This is the default value. Not using this entry.
        'ShrinkWrap Sigma': 5.,  # Default value
        'ShrinkWrap Decay Rate': 30,  # Default Value
        'ShrinkWrap Threshold Decay Ratio': 1.0,  # Default value.
        'ShrinkWrap Sigma Decay Ratio': 0.95,  # Default value.for
        'ShrinkWrap Filling Holes': True,
        'ShrinkWrap ConvexHull': False,
    },

    ###############################################################################################
    # 3rd Dict: Use error reduction
    ###############################################################################################
    {
        # Group 1: Set the algorithm properties
        'AlgName': 'ER',  # Algorithm name
        'IterNum': 1200,  # Iternation number
        # Because it is the Error Reduction, one does not need the other parameters.

        # Group 2: Set the initial support properties
        'InitSupport Type': 'Assigned',  # Initial support type
        # The following entry is for Type 'Assigned'. Because one derive the initial support
        # from auto-correlation, this is not used in this step in this chain.
        'InitSupport': 'Current Support',

        # Group 3: Set the initial density properties
        'InitDensity Type': "Assigned",  # Derive the density from the support.
        'InitDensity': 'Current Density',

        # Group 4: Set the shrink-wrap properties
        'ShrinkWrap Flag': False,  # Whether to use ShrinkWrap algorithm to update the support
        # Since the other parameters are either not used or not modified, I do not need to specify
        # their values again.

    }
]

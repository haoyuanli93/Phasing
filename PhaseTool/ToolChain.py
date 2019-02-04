"""
Usually, it is not enough to use a single algorithm to recover the phase.
This script put several algorithms together and implements some common
practice to make it easier for the users to use.
"""
from PhaseTool.AlterProj import CpuAlterProj


class AlterProjChain(CpuAlterProj):
    def __init__(self, device='cpu'):

        self.device = device
        # Set the device parameters since later on, one might have gpu version.
        super(CpuAlterProj, self).__init__()

        self.execute_sequence = {}

    def initialize(self,):
        pass

    def _algorithm_sequence_parser(self):
        pass

    def set_algorithm_sequence(self):
        pass

    def use_default_algorithm_sequence(self):
        pass

    def execute_algorithm_sequence(self,):
        pass

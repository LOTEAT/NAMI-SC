'''
Author: LOTEAT
Date: 2023-06-19 14:17:18
'''

import numpy as np
import torch
from ..builder import PIPELINES

@PIPELINES.register_module()
class CalculateSkelTransf:
    """Calculate skeletal transformation
    Args:
        keys (Sequence[str]): Required keys to be converted.
    """
    def __init__(self, enable=True, **kwargs):
        self.enable = enable

    def __call__(self, results):
        """
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.enable:
            smpl_pose = results['smpl_pose']
            joints = results['joints']
            parents = results['parents']
            # calculate the skeleton transformation
            smpl_pose = smpl_pose.reshape(-1, 3)
            A = get_rigid_transformation(smpl_pose, joints, parents)
            results['A'] = A
        return results

    def __repr__(self):
        return '{}:calculate the skeletal transformation'.format(
            self.__class__.__name__)

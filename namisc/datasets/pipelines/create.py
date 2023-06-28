'''
Author: LOTEAT
Date: 2023-06-18 22:13:47
'''
import torch
from ..builder import PIPELINES

@PIPELINES.register_module()
class DeleteUseless:
    """delete useless params
    Args:
        keys (Sequence[str]): Required keys to be converted.
    """
    def __init__(self, enable=True, keys=[], **kwargs):
        self.enable = enable
        self.keys = keys

    def __call__(self, results):
        """get viewdirs
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.enable:
            for k in self.keys:
                if k in results:
                    del results[k]
        return results

    def __repr__(self):
        return '{}:delete useless params'.format(self.__class__.__name__)


@PIPELINES.register_module()
class SampleData:
    """sample image from dataset
    Args:
        keys (Sequence[str]): Required keys to be converted.
    """
    def __init__(self, enable=True, **kwargs):
        self.enable = enable
        self.kwargs = kwargs

    def __call__(self, results):
        """BatchSlice
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.enable:
            idx = results['idx']
            results['data'] = results['data'][idx]

        return results

    def __repr__(self):
        return '{}:one data'.format(
            self.__class__.__name__)
        
@PIPELINES.register_module()
class GetDecoderData:
    """sample image from dataset
    Args:
        keys (Sequence[str]): Required keys to be converted.
    """
    def __init__(self, enable=True, is_test=False, **kwargs):
        self.enable = enable
        self.is_test = is_test
        self.kwargs = kwargs

    def __call__(self, results):
        """BatchSlice
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.enable:
            if self.is_test:
                device = results['data'].device
                results['target'] = self.kwargs['start_idx'] * torch.ones([1], dtype=torch.long).to(device)
                results['target_y'] = results['data']
            else:
                results['target'] = results['data'][:-1] # remove last one
                results['target_y'] = results['data'][1:] # remove first one
        return results

    def __repr__(self):
        return '{}:one data'.format(
            self.__class__.__name__)
        
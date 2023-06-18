'''
Author: LOTEAT
Date: 2023-06-18 22:13:47
'''
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

'''
Author: LOTEAT
Date: 2023-06-19 14:45:47
'''
import importlib
from ..builder import PIPELINES

@PIPELINES.register_module()
class TorchCall:
    def __init__(self, enable=True, mappings={}, torch_lib_path=None, torch_fun_name=None, **kwargs):
        self.enable = enable
        self.torch_lib_path = torch_lib_path
        self.torch_fun_name = torch_fun_name
        self.mappings = mappings
        self.kwargs = kwargs

    def __call__(self, results):
        """
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.enable:
            torch_module = importlib.import_module(self.torch_lib_path)
            torch_fun = getattr(torch_module, self.torch_fun_name)
            params = self.kwargs
            for map_key, param_key in self.mappings.items():
                params[param_key] = results[map_key]
            results['torch_call_res'] = torch_fun(params)
        return results
    

    def __repr__(self):
        return '{}:call torch function'.format(
            self.__class__.__name__)
    
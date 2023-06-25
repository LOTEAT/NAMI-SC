'''
Author: LOTEAT
Date: 2023-06-19 11:55:41
'''
from mmcv.runner import EpochBasedRunner
import time
import torch

class DeepSCTrainRunner(EpochBasedRunner):
    """DeepSCTrainRunner."""
    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, train_mode=False, **kwargs)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')


class DeepSCTestRunner(EpochBasedRunner):
    """DeepSCTestRunner."""
    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, train_mode=False, **kwargs)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

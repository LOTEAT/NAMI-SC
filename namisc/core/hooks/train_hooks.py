'''
Author: LOTEAT
Date: 2023-06-19 11:55:41
'''

import os
import numpy as np
from mmcv.runner.hooks import HOOKS, Hook
from mmcv.runner.hooks.lr_updater import LrUpdaterHook



@HOOKS.register_module()
class OccupationHook(Hook):
    """GPU source occupation hook GPU cards are fucking hard to queue recently
    Don't blame on me, I need only one card."""
    def __init__(self):
        self.first_run = True

    def func(self, runner):
        """OccupationHook func."""
        flag_folder = os.path.join(runner.work_dir, 'delete_me_to_stop')
        if self.first_run:
            os.makedirs(flag_folder, exist_ok=True)
            self.first_run = False
        else:
            # deldete that folder if you want to stop
            if not os.path.exists(flag_folder):
                print('Stop now!!!', flush=True)
                exit(0)

    def after_train_iter(self, runner):
        """OccupationHook after_train_iter."""
        self.func(runner)

    def after_val_iter(self, runner):
        """OccupationHook after_train_iter."""
        self.func(runner)


@HOOKS.register_module()
class MipLrUpdaterHook(LrUpdaterHook):
    """MipLrUpdaterHook."""
    def __init__(self,
                 lr_init,
                 lr_final,
                 max_steps,
                 lr_delay_steps=0,
                 lr_delay_mult=1,
                 **kwargs):
        super().__init__(**kwargs)
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.max_steps = max_steps
        self.lr_delay_steps = lr_delay_steps
        self.lr_delay_mult = lr_delay_mult

    def get_lr(self, runner, base_lr):
        """get_lr."""
        step = runner.epoch if self.by_epoch else runner.iter
        if self.lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = self.lr_delay_mult + (
                1 - self.lr_delay_mult) * np.sin(
                    0.5 * np.pi * np.clip(step / self.lr_delay_steps, 0, 1))
        else:
            delay_rate = 1.
        t = np.clip(step / self.max_steps, 0, 1)
        log_lerp = np.exp(
            np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t)
        return delay_rate * log_lerp

'''
Author: LOTEAT
Date: 2023-06-19 11:55:41
'''

import os

import imageio
import numpy as np
import torch
from mmcv.runner import get_dist_info
from mmcv.runner.hooks import HOOKS, Hook

# from .utils import calculate_ssim, img2mse, mse2psnr, to8b


@HOOKS.register_module()
class SetValPipelineHook(Hook):
    """pass val dataset's pipeline to network."""
    def __init__(self, valset=None):
        self.val_pipeline = valset.pipeline

    def before_run(self, runner):
        """only run once."""
        runner.model.module.set_val_pipeline(self.val_pipeline)
        del self.val_pipeline

@HOOKS.register_module()
class ValidateHook(Hook):
    """在测试集上计算ssim psnr指标 保存图片."""
    def __init__(self, save_folder='validation'):
        self.save_folder = save_folder

    def after_val_iter(self, runner):
        """ValidateHook."""
        pass

@HOOKS.register_module()
class CalElapsedTimeHook(Hook):
    """calculate average elapsed_time in val step."""
    def __init__(self, cfg=None):
        self.cfg = cfg

    def after_val_iter(self, runner):
        """after_val_iter."""
        rank, _ = get_dist_info()
        if rank == 0:
            if 'elapsed_time' in runner.outputs:
                elapsed_time_list = runner.outputs['elapsed_time']
            else:
                elapsed_time_list = []
            if len(elapsed_time_list) == 0: return

            #calculate average elapsed time
            average_elapsed_time = 1000 * sum(elapsed_time_list) / len(
                elapsed_time_list)

            metrics = 'On testset, elapsed_time is {:7.2f} ms'.format(
                average_elapsed_time)
            runner.logger.info(metrics)
            # exit(0)

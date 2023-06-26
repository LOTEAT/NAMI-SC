'''
Author: LOTEAT
Date: 2023-06-19 11:55:41
'''

import json
import os

from mmcv.runner import get_dist_info
from mmcv.runner.hooks import HOOKS, Hook
from .utils import calculate_metrics
import numpy as np

@HOOKS.register_module()
class TextTestHook(Hook):
    """In test phase, calculate metrics over all testset.

    ndown: multiscales for mipnerf, set to 0 for others
    """
    def __init__(self,
                 ndown=1,
                 metric_names=[],
                 dump_json=False):
        self.ndown = ndown
        self.metric_names = metric_names
        self.dump_json = dump_json


    def before_val_epoch(self, runner):
        """init list."""
        self.words = []
        self.targets = []
        self.metrics = {name: [] for name in self.metric_names}

    def after_val_iter(self, runner):
        """after_val_iter."""
        rank, _ = get_dist_info()
        if rank == 0:
            words = runner.outputs['words']
            targets = runner.outputs['targets']
            self.words.extend(words)
            self.targets.extend(targets)

    def after_val_epoch(self, runner):
        """after_val_epoch."""
        rank, _ = get_dist_info()
        if rank == 0:
            metrics = 'In test phase on whole testset: \n  '
            data = {'words': self.words, 'targets': self.targets}
            for name in self.metric_names:
                score = np.array(calculate_metrics(data, name)).mean()
                metrics += f'Epoch {runner._epoch} For metric {name}, score is {score}. \n'
                self.metrics[name].append(score)
                mean_score = np.array(self.metrics[name]).mean()
                metrics += f' For metric {name}, current mean score is {mean_score}. \n'
            runner.logger.info(metrics)

            # if self.dump_json:
            #     filename = os.path.join(runner.work_dir, self.save_folder,
            #                             'test_results.json')
            #     with open(filename, 'w') as f:
            #         json.dump(
            #             {
            #                 'results': metrics,
            #                 'psnrs': self.psnr,
            #                 'ssims': self.ssim
            #             }, f)
            '''
                in mmcv's EpochBasedRunner, only 'after_train_epoch' epoch will be updated
                but in our test phase, we only want to run ('val', 1),
                so we need to update runner_epoch additionally
            '''
            runner._epoch += 1

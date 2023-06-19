'''
Author: LOTEAT
Date: 2023-05-31 18:37:44
'''
import torch.nn as nn
from .base import BaseTranseiver
from .. import builder
from ..builder import TRANSCEIVER


@TRANSCEIVER.register_module()
class DeepSCTranseiver(BaseTranseiver):
    """There are 3 kinds of forward mode for Network:

    1. 'train': phase=='train' and use 'train_step()' to forward, input a batch of rays
    2. 'val': phase=='train' and 'val_step()' to forward, input all testset's poses&images in one 'val_step()'
    3. 'test': phase=='test' and 'test_step()' to forward, input all testset one by one
    """
    def __init__(self, cfg, se=None, sd=None, channel=None, cd=None, ce=None):
        super().__init__()

        self.phase = cfg.get('phase', 'train')
        self.se = builder.build_se(se)
        self.sd = builder.build_sd(sd)
        self.ce = builder.build_ce(ce)
        self.cd = builder.build_cd(cd)
        self.channel = builder.build_channel(channel)

    def forward(self, data, is_test=False):
        pass
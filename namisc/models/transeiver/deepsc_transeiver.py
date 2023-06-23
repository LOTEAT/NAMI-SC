'''
Author: LOTEAT
Date: 2023-05-31 18:37:44
'''
import torch.nn as nn
import torch
from .base import BaseTranseiver
from .. import builder
from ..builder import TRANSCEIVER
from ..utils import SparseCategoricalCrossentropyLoss

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
        data = self.se(data)
        data = self.ce(data)
        data = self.channel(data)
        data = self.cd(data)
        data = self.sd(data)
        return data
        
    
    def train_step(self, data, optimizer, **kwargs):
        device = data['data'].device
        target_real = data['target_y']
        tgt_size = data['target'].size(1)
        look_ahead_mask = 1 - torch.tril(torch.ones(tgt_size, tgt_size))
        look_ahead_mask = look_ahead_mask.to(device)
        data['combined_mask'] = torch.max(data['target_padding_mask'], look_ahead_mask)
        ret = self.forward(data, is_test=False)
        loss = SparseCategoricalCrossentropyLoss(target_real, ret['data'])
        log_vars = {'loss': loss.item()}
        outputs = {
            'loss': loss,
            'log_vars': log_vars,
            'num_samples': data['data'].shape[0]
        }
        return outputs
    
    def val_step(self, data, **kwargs):
        return super().val_step(data, **kwargs)
    
    def test_step(self, data, **kwargs):
        pass
    
    def set_val_pipeline(self, func):
        self.val_pipeline = func
        return
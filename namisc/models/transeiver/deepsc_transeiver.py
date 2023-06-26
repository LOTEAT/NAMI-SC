'''
Author: LOTEAT
Date: 2023-05-31 18:37:44
'''
import torch.nn as nn
import torch
from .base import BaseTranseiver
from .. import builder
from ..builder import TRANSCEIVER
from ...utils import sparse_categorical_cross_entropy
from ..utils import get_look_ahead_mask, get_padding_mask

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
        self.max_length = cfg.get('max_length', 35)
        self.se = builder.build_se(se)
        self.sd = builder.build_sd(sd)
        self.ce = builder.build_ce(ce)
        self.cd = builder.build_cd(cd)
        self.channel = builder.build_channel(channel)
        self.words = [] 
        self.targets = []

    def forward(self, data, is_test=False):
        data = self.se(data)
        data = self.ce(data)
        data = self.channel(data)
        data = self.cd(data)
        data = self.sd(data)
        return data
        
    
    def train_step(self, data, optimizer, **kwargs):
        device = data['data'].device
        tgt_size = data['target'].size(1)
        look_ahead_mask = get_look_ahead_mask(tgt_size).to(device)
        data['target_padding_mask'] = torch.max(data['target_padding_mask'], look_ahead_mask)
        ret = self.forward(data, is_test=False)
        loss = sparse_categorical_cross_entropy(data['target_y'], ret['data'])
        log_vars = {'loss': loss.item()}
        outputs = {
            'loss': loss,
            'log_vars': log_vars,
            'num_samples': data['data'].shape[0]
        }
        return outputs
    
    def val_step(self, data, optimizer, **kwargs):
        return self.test_step(data, optimizer, **kwargs)
    
    def test_step(self, data, optimizer, **kwargs):
        device = data['data'].device
        tgt_size = data['target'].size(1)    
        look_ahead_mask = get_look_ahead_mask(tgt_size).to(device)
        data['target_padding_mask'] = torch.max(data['target_padding_mask'], look_ahead_mask)
        data = self.cd(self.channel(self.ce(self.se(data))))
        cd_output = data['data'].clone()
        for _ in range(self.max_length):
            data['data'] = cd_output.clone()
            ret = self.sd(data)
            prediction = ret['data'][:, -1:, :]  # (batch_size, 1, vocab_size)
            predicted_idx = torch.argmax(prediction, dim=-1).long()
            outputs = torch.cat([data['target'], predicted_idx], dim=-1)
            look_ahead_mask = get_look_ahead_mask(outputs.size(1)).to(device)
            data['target_padding_mask'] = get_padding_mask(outputs).to(device)
            data['target_padding_mask']= torch.max(data['target_padding_mask'], look_ahead_mask)
            data['target'] = outputs

    
        sentences = outputs.cpu().numpy().tolist()
        words = list(map(kwargs.get('extra_func'), sentences))

        target_sentences = data['target'].cpu().numpy().tolist()
        targets = list(map(kwargs.get('extra_func'), target_sentences))
        
        outputs = {
            'words': words,
            'targets': targets, 
        }
        return outputs
    
    def set_val_pipeline(self, func):
        self.val_pipeline = func
        return
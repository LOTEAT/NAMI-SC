'''
Author: LOTEAT
Date: 2023-05-31 18:37:44
'''
import torch.nn as nn
import torch
from .base import BaseTranseiver
from .. import builder
from ..builder import TRANSCEIVER
from ...utils import SparseCategoricalCrossentropyLoss


def create_padding_mask(seq):
    seq = torch.eq(seq, 0).float()
    return seq.unsqueeze(1).unsqueeze(2) 

def create_look_ahead_mask(size):
    mask = 1 - torch.tril(torch.ones(size, size))
    return mask 

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
        look_ahead_mask = 1 - torch.tril(torch.ones(tgt_size, tgt_size))
        look_ahead_mask = look_ahead_mask.to(device)
        data['combined_mask'] = torch.max(data['target_padding_mask'], look_ahead_mask)
        ret = self.forward(data, is_test=False)
        loss = SparseCategoricalCrossentropyLoss(data['target_y'], ret['data'])
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
        look_ahead_mask = 1 - torch.tril(torch.ones(tgt_size, tgt_size))
        look_ahead_mask = look_ahead_mask.to(device)
        data['combined_mask'] = torch.max(data['target_padding_mask'], look_ahead_mask)
        ret = self.cd(self.channel(self.ce(self.se(data))))
        
        cd_output = ret['data'].clone()
        predictions = []
        for _ in range(35):
            ret['data'] = cd_output.clone()
            ret = self.sd(ret)
            prediction = ret['data'][:, -1:, :]  # (batch_size, 1, vocab_size)
            predictions.append(prediction)
            predicted_idx = torch.argmax(prediction, dim=-1).long()
            outputs = torch.cat([data['target'], predicted_idx], dim=-1)
            look_ahead_mask = create_look_ahead_mask(outputs.size(1)).to(device)
            data['target_padding_mask'] = create_padding_mask(outputs).to(device)
            data['combined_mask']= torch.max(data['target_padding_mask'], look_ahead_mask)
            data['target'] = outputs

        predictions = torch.cat(predictions, dim=1)
        sentences = outputs.cpu().numpy().tolist()
        words = list(map(kwargs.get('extra_func'), sentences))

        target_sent = data['target'].cpu().numpy().tolist()
        targets = list(map(kwargs.get('extra_func'), target_sent))
        
        outputs = {
            'words': words,
            'targets': targets, 
        }
        return outputs
    
    def set_val_pipeline(self, func):
        self.val_pipeline = func
        return
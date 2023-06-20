'''
Author: LOTEAT
Date: 2023-05-31 18:37:44
'''
import torch.nn as nn
import torch
from .base import BaseTranseiver
from .. import builder
from ..builder import TRANSCEIVER
from ..utils import SparseCategoricalCrossentropyLoss, create_look_ahead_mask, create_padding_mask


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
        data = self.se(data['data'], data['enc_padding_mask'])
        data = self.ce(data)
        data = self.channel(data)
        data = self.cd(data)
        data = self.sd(data)
        return data
        
    
    def train_step(self, data, optimizer, **kwargs):
        device = data['data'].device
        target_input = data['target'][:, :-1]
        target_real = data['target'][:, 1:]
        
        data['enc_padding_mask'] = create_padding_mask(data).to(device)
        data['dec_padding_mask'] = create_padding_mask(data).to(device)
        data['look_ahead_mask'] = create_look_ahead_mask(data['target'].size(1)).to(device)
        data['dec_target_padding_mask'] = create_padding_mask(data['target'])
        data['combined_mask'] = torch.max(data['dec_target_padding_mask'], data['look_ahead_mask'])
        ret = self.forward(data, is_test=False)

        # # rgb: fine network's out, coarse_rgb: coarse's
        # img_loss = img2mse(ret['rgb'], data['target_s'])
        # psnr = mse2psnr(img_loss)
        # loss = img_loss

        # if 'coarse_rgb' in ret:
        #     coarse_img_loss = img2mse(ret['coarse_rgb'], data['target_s'])
        #     loss = loss + coarse_img_loss
        #     coarse_psnr = mse2psnr(coarse_img_loss)

        # log_vars = {'loss': loss.item(), 'psnr': psnr.item()}
        # outputs = {
        #     'loss': loss,
        #     'log_vars': log_vars,
        #     'num_samples': ret['rgb'].shape[0]
        # }
        # return outputs
        
        
    # tar_inp = target[:, :-1]  # exclude the last one
    # tar_real = target[:, 1:]  # exclude the first one

    # enc_padding_mask, combined_mask, dec_padding_mask = create_masks(data, tar_inp)

    # optim_net.zero_grad()
    # optim_mi.zero_grad()

    # # Forward pass
    # predictions, channel_enc_output, received_channel_enc_output = transceiver(
    #     data, tar_inp, channel=channel, n_std=n_std,
    #     enc_padding_mask=enc_padding_mask,
    #     combined_mask=combined_mask, dec_padding_mask=dec_padding_mask
    # )
    # # Compute loss
    # loss_error = criterion(tar_real, predictions)
    # loss = loss_error
    
    # if use_mine:
    #     joint, marginal = sample_batch(channel_enc_output, received_channel_enc_output)
    #     mi_lb, _, _ = mutual_information(joint, marginal, mine_net)
    #     loss_mine = -mi_lb
    #     loss += 0.05 * loss_mine

    # # Compute gradients and update network parameters
    # loss.backward()
    # optim_net.step()

    # if use_mine:
    #     # Compute gradients and update MI estimator parameters
    #     optim_mi.zero_grad()
    #     loss_mine.backward()
    #     optim_mi.step()

    # mi_numerical = 2.20  # Placeholder value, update with actual value

    # return loss, None, mi_numerical

        return super().train_step(data, optimizer, **kwargs)
    
    def val_step(self, data, **kwargs):
        return super().val_step(data, **kwargs)
    
    def test_step(self, data, **kwargs):
        pass
    
    def set_val_pipeline(self, func):
        self.val_pipeline = func
        return
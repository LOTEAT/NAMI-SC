'''
Author: LOTEAT
Date: 2023-06-18 20:48:31
'''
import torch.nn as nn
import torch
from ..builder import SE
from .base import BaseSE
from ..utils import normalize_data, enframe, deframe

@SE.register_module()
class DeepSCSSemanticEncoder(BaseSE):
    def __init__(self, frame_length, stride_length, num_frame, sem_enc_outdims):
        super(DeepSCSSemanticEncoder, self).__init__()
        self.num_frame = num_frame
        self.frame_length = frame_length
        self.stride_length = stride_length
        self.sem_enc_outdims = sem_enc_outdims

        self.layers = nn.ModuleList([
            nn.Conv2d(3, self.outdims[0], kernel_size=(5, 5), stride=(2, 2), padding=2, bias=False),
            nn.BatchNorm2d(self.outdims[0]),
            nn.ReLU(),
            nn.Conv2d(self.outdims[0], self.outdims[1], kernel_size=(5, 5), padding=2, stride=(2, 2), bias=False),
            nn.BatchNorm2d(self.outdims[1]),
            nn.ReLU(),
        ])


        self.resnet_modules = nn.ModuleList()
        for module_count, outdim in enumerate(self.sem_enc_outdims[2:]):
            self.resnet_modules.append(SEResNet(outdim))

    def forward(self, data):
        # Preprocessing _input
        normalized_data, mean, var = normalize_data(data)  # Assuming wav_norm is defined separately
        normalized_data = enframe(normalized_data, self.num_frame, self.frame_length, self.stride_length)
        normalized_data = torch.unsqueeze(normalized_data, dim=1)

        ###################### Semantic Encoder ######################
        output = self.layers(normalized_data)

        return output, mean, var


class SEResNet(nn.Module):
    def __init__(self, out_dim):
        super(SEResNet, self).__init__()
        # Define layers of the SEResNet module

    def forward(self, x):
        # Forward pass logic of the SEResNet module
        return x

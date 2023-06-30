'''
Author: LOTEAT
Date: 2023-06-18 20:48:31
'''
import torch.nn as nn
from ..builder import SE
from .base import BaseSE
from ..utils import normalize_data
from ..transformer.base.encoder import TransformerEncoder
from ..transformer.attention.position import PositionalEncoding
from ..transformer.attention.embedding import Embeddings

@SE.register_module()
class DeepSCSSemanticEncoder(BaseSE):
    def __init__(self, frame_length, stride_length, num_frame, sem_enc_outdims):
        super(DeepSCSSemanticEncoder, self).__init__()
        self.num_frame = num_frame
        self.frame_length = frame_length
        self.stride_length = stride_length
        self.sem_enc_outdims = sem_enc_outdims

        # Define layers
        self.conv1 = nn.Conv2d(1, self.sem_enc_outdims[0], kernel_size=(3, 3), stride=(2, 2))
        self.conv2 = nn.Conv2d(self.sem_enc_outdims[0], self.sem_enc_outdims[1], kernel_size=(3, 3), stride=(2, 2))
        self.resnet_modules = nn.ModuleList()
        for module_count, outdim in enumerate(self.sem_enc_outdims[2:]):
            self.resnet_modules.append(SEResNet(outdim))

    def forward(self, data):
        # Preprocessing _input
        normalized_data, mean, var = normalize_data(data)  # Assuming wav_norm is defined separately
        _input = enframe(_input, self.num_frame, self.frame_length, self.stride_length)
        _input = torch.unsqueeze(_input, dim=1)

        ###################### Semantic Encoder ######################
        _output = self.conv1(_input)
        _output = F.relu(_output)
        _output = self.conv2(_output)
        _output = F.relu(_output)
        for module_count, resnet_module in enumerate(self.resnet_modules):
            _output = resnet_module(_output)
            _output = F.relu(_output)

        return _output, batch_mean, batch_var


class SEResNet(nn.Module):
    def __init__(self, out_dim):
        super(SEResNet, self).__init__()
        # Define layers of the SEResNet module

    def forward(self, x):
        # Forward pass logic of the SEResNet module
        return x

'''
Author: LOTEAT
Date: 2023-06-16 16:38:17
'''
from torch import nn
from ..utils.powernorm import PowerNorm
from ..builder import CE
from .base import BaseCE

@CE.register_module()
class DeepSCChannelEncoder(BaseCE):
    def __init__(self, d_model=128):
        super(DeepSCChannelEncoder, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 16),
            PowerNorm()
        )

    def forward(self, data):
        out = self.layers(data['data'])
        data['data'] = out
        return data

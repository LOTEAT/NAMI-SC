'''
Author: LOTEAT
Date: 2023-06-16 16:38:17
'''
from torch import nn
from ..builder import MINE
from .base import BaseMINE

@MINE.register_module()
class DeepSCMine(BaseMINE):
    def __init__(self, hidden_size=10):
        super(DeepSCMine, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.apply(self.init_weights)
        
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight.data, mean=0.0, std=0.02)
            nn.init.constant_(module.bias.data, 0)
        
    def forward(self, inputs):
        output = self.layers(inputs)
        return output

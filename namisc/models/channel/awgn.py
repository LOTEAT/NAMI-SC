'''
Author: LOTEAT
Date: 2023-06-19 13:24:30
'''

from .base import BaseChannel
import torch
from ..builder import CHANNEL

@CHANNEL.register_module()
class Awgn(BaseChannel):
    def __init__(self):
        super(Awgn, self).__init__()

    def forward(self, data, n_std):
        y = data + torch.randn_like(data) * n_std
        return y
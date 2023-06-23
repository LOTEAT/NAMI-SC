'''
Author: LOTEAT
Date: 2023-06-19 13:24:30
'''

from .base import BaseChannel
import torch
from ..builder import CHANNEL

@CHANNEL.register_module()
class Awgn(BaseChannel):
    def __init__(self, n_std):
        super(Awgn, self).__init__()
        self.n_std = n_std

    def forward(self, data):
        data['data'] = data['data'] + torch.randn_like(data['data']) * self.n_std
        return data
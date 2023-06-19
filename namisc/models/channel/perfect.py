'''
Author: LOTEAT
Date: 2023-06-18 20:21:21
'''
from .base import BaseChannel
from ..builder import CHANNEL

@CHANNEL.register_module()
class PerfectChannel(BaseChannel):
    def __init__(self):
        pass

    def forward(self, data):
        # Nothing to be changed
        return data
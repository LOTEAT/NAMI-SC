'''
Author: LOTEAT
Date: 2023-06-19 13:24:39
'''
from .base import BaseChannel
from ..builder import CHANNEL

@CHANNEL.register_module()
class Fading(BaseChannel):
    def __init__(self):
        pass

    def forward(self, **kwargs):
        # TODO
        raise NotImplementedError
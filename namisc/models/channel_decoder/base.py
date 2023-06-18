'''
Author: LOTEAT
Date: 2023-06-18 20:51:15
'''
# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from torch import nn
from ..builder import CD


@CD.register_module()
class BaseCD(nn.Module, metaclass=ABCMeta):
    def __init__(self, **kwarg):
        super().__init__()

    @abstractmethod
    def forward(self, inputs):
        raise NotImplementedError

'''
Author: LOTEAT
Date: 2023-06-18 20:51:15
'''
# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import torch
from torch import nn

from ..builder import CE


@CE.register_module()
class BaseCE(nn.Module, metaclass=ABCMeta):
    def __init__(self, **kwarg):
        super().__init__()

    @abstractmethod
    def forward(self, inputs):
        raise NotImplementedError

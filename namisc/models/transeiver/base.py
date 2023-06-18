'''
Author: LOTEAT
Date: 2023-06-18 21:58:17
'''
# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from torch import nn

class BaseTranseiver(nn.Module, metaclass=ABCMeta):
    def __init__(self, **kwarg):
        super().__init__()

    @abstractmethod
    def train_step(self, data, optimizer, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def val_step(self, data, **kwargs):
        raise NotImplementedError

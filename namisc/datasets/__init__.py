'''
Author: LOTEAT
Date: 2023-06-19 13:20:13
'''
from .builder import build_dataset
from .europarl_dataset import EuroparlDataset

__all__ = [
    'build_dataset',
    'EuroparlDataset'
]

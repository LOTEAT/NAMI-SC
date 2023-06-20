'''
Author: LOTEAT
Date: 2023-06-20 20:05:11
'''

import numpy as np
import torch

def snr2noise(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)
    return noise_std

def create_padding_mask(seq):
    seq = torch.eq(seq, 0).float()
    return seq.unsqueeze(1).unsqueeze(2) 

def create_look_ahead_mask(size):
    mask = 1 - torch.tril(torch.ones(size, size))
    return mask 

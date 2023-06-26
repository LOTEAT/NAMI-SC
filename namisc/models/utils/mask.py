'''
Author: LOTEAT
Date: 2023-06-26 16:13:14
'''
import torch 

def get_padding_mask(seq):
    seq = torch.eq(seq, 0).float()
    return seq.unsqueeze(1).unsqueeze(2) 

def get_look_ahead_mask(size):
    mask = 1 - torch.tril(torch.ones(size, size))
    return mask 
'''
Author: LOTEAT
Date: 2023-06-30 16:09:58
'''
import torch

def normalize_data(data):
    mean = torch.mean(data, dim=-1, keepdim=True)
    var = torch.var(data, dim=-1, keepdim=True)
    normalized_data = (data - mean) / torch.sqrt(var)
    return normalized_data, mean, var

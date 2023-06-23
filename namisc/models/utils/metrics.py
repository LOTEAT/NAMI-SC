'''
Author: LOTEAT
Date: 2023-06-20 19:30:25
'''
import torch
from torch import nn



def SparseCategoricalCrossentropyLoss(real, pred, ignore_index=0):
    loss_object = nn.CrossEntropyLoss(reduction='none')
    mask = real != ignore_index
    bs = pred.shape[0]
    loss_ = loss_object(pred.view(-1, 22234), real.contiguous().view(-1))
    loss_ = loss_.view(bs, -1)
    loss_ *= mask.float()
    return torch.mean(loss_)






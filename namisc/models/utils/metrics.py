'''
Author: LOTEAT
Date: 2023-06-20 19:30:25
'''
import torch
from torch import nn

class SparseCategoricalCrossentropyLoss(nn.Module):
    def __init__(self, ignore_index=0):
        super(SparseCategoricalCrossentropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.loss_object = nn.CrossEntropyLoss(reduction='none')

    def forward(self, real, pred):
        mask = real != self.ignore_index
        bs = pred.shape[0]

        loss_ = self.loss_object(pred.view(-1, 22234), real.contiguous().view(-1))
        loss_ = loss_.view(bs, -1)
        loss_ *= mask.float()

        return torch.mean(loss_)






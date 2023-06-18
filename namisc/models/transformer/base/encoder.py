'''
Author: LOTEAT
Date: 2023-06-18 00:03:49
'''

from torch import nn
from ..attention.attention import MultiHeadedAttention
from ..attention.sublayer import SublayerConnection
from ..attention.feedforward import PositionwiseFeedForward

class TransformerEncoder(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, d_model, num_heads, dff, dropout):
        super(TransformerEncoder, self).__init__()
        self.self_attn = MultiHeadedAttention(num_heads, d_model)
        self.sublayer1 = SublayerConnection(size, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, dff)
        self.sublayer2 = SublayerConnection(size, dropout)
        self.size = size

    def forward(self, x, mask):
        attn_output, _ = self.self_attn(x, x, x, mask)  
        attn_output = self.sublayer1(x, attn_output)
        ffn_output = self.feed_forward(attn_output)
        ffn_output = self.sublayer2(attn_output, ffn_output)
        return ffn_output

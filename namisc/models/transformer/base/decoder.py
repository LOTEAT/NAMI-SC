'''
Author: LOTEAT
Date: 2023-06-18 00:04:05
'''
from torch import nn
from ..attention.attention import MultiHeadedAttention
from ..attention.sublayer import SublayerConnection
from ..attention.feedforward import PositionwiseFeedForward

class TransformerDecoder(nn.Module):
    """
    This is decoder leayer, which includes three layers,
    1. multihead,
    2. masked multihead
    3. feed forward
    """

    def __init__(self, size, d_model, num_heads, dff, drop_pro=0.1):
        super(TransformerDecoder, self).__init__()
        
        self.attention_layer1 = MultiHeadedAttention(num_heads, d_model)  # masked
        self.attention_layer2 = MultiHeadedAttention(num_heads, d_model)

        self.ffn = PositionwiseFeedForward(d_model, dff)
        self.sublayer1 = SublayerConnection(size, drop_pro)
        self.sublayer2 = SublayerConnection(size, drop_pro)
        self.sublayer3 = SublayerConnection(size, drop_pro)
        self.feed_forward = PositionwiseFeedForward(d_model, dff)
        self.size = size


    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        attn1, attn_weights1 = self.attention_layer1(x, x, x, look_ahead_mask)
        output1 = self.sublayer1(x, attn1)
        attn2, attn_weights2 = self.attention_layer2(
            output1, enc_output, enc_output, padding_mask
        )       
        output2 = self.sublayer2(output1, attn2)
        ffn_output = self.ffn(output2)
        output3 = self.sublayer3(output2, ffn_output)
        return output3, attn_weights1, attn_weights2

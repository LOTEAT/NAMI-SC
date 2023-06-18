'''
Author: LOTEAT
Date: 2023-06-18 20:19:49
'''

import torch.nn as nn
import math
from ..builder import SD
from .base import BaseSD
from ..transformer.base.decoder import TransformerDecoder
from ..transformer.attention.position import PositionalEncoding
from ..transformer.attention.embedding import Embeddings
class SemanticDecoder(BaseSD):
    """
    1. Output Embedding
    2. Positional Encoding
    3. N decoder layers
    """

    def __init__(
        self,
        num_layers,
        num_heads,
        d_model,
        dff,
        vocab_size,
        max_pos_encoding=512,
        sd_dropout=0.1,
        pos_dropout=0
    ):
        super(SemanticDecoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = Embeddings(d_model, vocab_size)
        self.pos_encoding = PositionalEncoding(d_model, pos_dropout, max_pos_encoding)

        self.dec_layers = nn.ModuleList([
            TransformerDecoder(d_model, d_model, num_heads, dff, sd_dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(sd_dropout)
        # prediction layer
        self.final_layer = nn.Linear(128, vocab_size)

    def forward(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attention_weights = {}
        x = self.embedding(x) 
        x = self.pos_encoding(x)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](
                x, enc_output, look_ahead_mask, padding_mask
            )

        attention_weights["decoder_layer{}_block1".format(i + 1)] = block1
        attention_weights["decoder_layer{}_block2".format(i + 1)] = block2
        x = self.final_layer(x)

        return x, attention_weights

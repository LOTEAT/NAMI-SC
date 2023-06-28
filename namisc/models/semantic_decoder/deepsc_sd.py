'''
Author: LOTEAT
Date: 2023-06-18 20:19:49
'''

import torch.nn as nn
from ..builder import SD
from .base import BaseSD
from ..transformer.base.decoder import TransformerDecoder
from ..transformer.attention.position import PositionalEncoding
from ..transformer.attention.embedding import Embeddings

@SD.register_module()
class DeepSCSemanticDecoder(BaseSD):
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
        super(DeepSCSemanticDecoder, self).__init__()

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
        self.final_layer = nn.Linear(d_model, vocab_size)

    def forward(self, data):
        x = self.embedding(data['target']) 
        x = self.pos_encoding(x)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x, _, _ = self.dec_layers[i](
                x, data['data'], data['target_padding_mask'], data['dec_padding_mask']
            )
        x = self.final_layer(x)
        data['data'] = x
        return data

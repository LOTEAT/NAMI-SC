import torch.nn as nn
import math
from ..builder import SE
from .base import BaseSE
from ..transformer.base.encoder import TransformerEncoder
from ..transformer.attention.position import PositionalEncoding

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

@SE.register_module()
class SemanticEncoder(BaseSE):
    def __init__(
        self,
        num_layers,
        num_heads,
        d_model,
        dff,
        vocab_size,
        max_pos_encoding=512,
        se_dropout=0.1,
        pos_dropout=0
    ):
        super(SemanticEncoder, self).__init__()
        self.d_model = d_model
        self.dff = dff
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.embedding = Embeddings(d_model, vocab_size)
        self.pos_encoding = PositionalEncoding(d_model, pos_dropout, max_pos_encoding)  
        self.dropout = nn.Dropout(se_dropout)
        self.encoder = nn.ModuleList([
            TransformerEncoder(d_model, d_model, num_heads, dff, se_dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask):
        self.pos_encoding = self.pos_encoding.to(x.device)
        # Embedding
        x = self.embedding(x)
        # positional Encoding
        x = self.pos_encoding(x)
        # Dropout
        x = self.dropout(x)
        # Encoder
        for i in range(self.num_layers):
            x = self.encoder[i](x, mask)
        return x
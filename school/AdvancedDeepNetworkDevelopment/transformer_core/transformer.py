import math
import torch

from torch import nn
from transformer_core.encoder import Encoder
from transformer_core.decoder import Decoder
from transformer_core.positional_encoding import PositionalEncoding

class Transformer(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, num_blocks:int = 6, d_model: int = 512, num_heads: int = 8):
        super().__init__()
        self.encoder = Encoder(num_blocks, d_model, num_heads)
        self.decoder = Decoder(num_blocks, d_model, num_heads)
        self.pe = PositionalEncoding(d_model)
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.final_linear = nn.Linear(d_model, tgt_vocab_size)

        self.d_model = d_model

    def forward(self, x: torch.Tensor, y:torch.Tensor, padding_mask: torch.Tensor, look_ahead_mask: torch.Tensor) -> torch.Tensor:
        x = self.src_embed(x) * math.sqrt(self.d_model)
        x = self.pe(x)
        x = self.encoder(x, padding_mask)

        y = self.tgt_embed(y) * math.sqrt(self.d_model)
        y = self.pe(y)

        output = self.decoder(y, x_encoder=x, look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)
        logits = self.final_linear(output)
        return logits

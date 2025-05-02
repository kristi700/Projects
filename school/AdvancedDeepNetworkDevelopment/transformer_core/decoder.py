import torch

from torch import nn
from transformer_core.attention import MultiHeadedAttention
from transformer_core.feed_forward import FeedForwardBlock

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int = 512, num_heads: int = 8):
        super().__init__()
        self.self_attention = MultiHeadedAttention(d_model, num_heads)
        self.cross_attention = MultiHeadedAttention(d_model, num_heads)
        self.feed_forward = FeedForwardBlock(d_model)

        # Separate LayerNorms as elementwise=True
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, x_encoder: torch.Tensor, look_ahead_mask:torch.Tensor, padding_mask:torch.Tensor) -> torch.Tensor:
        attn_output = self.self_attention(query=x, key=x, value=x, mask=look_ahead_mask)
        x = self.layer_norm1(attn_output + x)
        attn_output = self.cross_attention(query=x, key=x_encoder, value=x_encoder, mask=padding_mask)
        x = self.layer_norm2(attn_output + x)
        ff_output = self.feed_forward(x)
        x = self.layer_norm3(ff_output + x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, num_blocks:int = 6, d_model: int = 512, num_heads: int = 8):
        super().__init__()
        self.decoder_blocks = nn.ModuleList([DecoderBlock(d_model, num_heads) for i in range(num_blocks)])

    def forward(self, x: torch.Tensor, x_encoder: torch.Tensor, look_ahead_mask:torch.Tensor, padding_mask:torch.Tensor) -> torch.Tensor:
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, x_encoder, look_ahead_mask, padding_mask)
        return x


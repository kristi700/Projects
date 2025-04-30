import torch

from torch import nn

from transformer_core.feed_forward import FeedForwardBlock
from transformer_core.attention import MultiHeadedAttention
from transformer_core.positional_encoding import PositionalEncoding
# TODO - do we need dk dv?

class EncoderBlock(nn.Module):
    """
    TODO
    """
    def __init__(self, d_model: int = 512, num_heads: int = 8):
        super().__init__()
        self.mha = MultiHeadedAttention(d_model, num_heads)
        self.feed_forward = FeedForwardBlock(d_model)

        # Separate LayerNorms as elementwise=True
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        TODO
        """
        attn_output = self.mha(query=x, key=x, value=x, mask=mask)
        x = self.layer_norm1(attn_output + x)
        ff_output = self.feed_forward(x)
        x=self.layer_norm2(ff_output+x)
        return x
    
class Encoder(nn.Module):
    """
    TODO
    """
    def __init__(self, num_blocks:int = 6, d_model: int = 512, num_heads: int = 8):
        super().__init__()
        self.encoder_blocks = nn.ModuleList([EncoderBlock(d_model, num_heads) for i in range(num_blocks)])

    def forward(self, x: torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, mask)
        return x


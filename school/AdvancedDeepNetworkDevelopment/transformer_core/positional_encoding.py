import torch

from torch import nn

class PositionalEncoding(nn.Module):
  def __init__(self, d_model: int, dropout: float = 0.1, max_length: int = 5000):
    """
    Args:
        - d_model: dimension of embeddings
        - dropout: randomly zeroes-out some of the input
        - max_length: max sequence length
    Output:
        - None
    """
    super().__init__()
    self.dropout= nn.Dropout(p=dropout)
    position=torch.arange(max_length).unsqueeze(1)
    div_term = 1 / (10000 ** (torch.arange(0, d_model, 2) / d_model)) # NOTE - allegedy not the most stable way of implementing it
    angles = position * div_term

    pe = torch.zeros(max_length, d_model)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)

    self.max_length = max_length
    self.register_buffer('pe', pe.unsqueeze(0))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        - x: input embeddings of shape (batch_size, seq_len, d_model)
    Output:
        - embeddings with pe added of shape (batch_size, seq_len, d_model)
    """
    seq_len = x.shape[1]
    assert self.max_length >= seq_len, f"Sequence length cannot be bigger than max_length, {seq_len} > {self.max_length}"
    x = x + self.pe[:, :seq_len, :]
    return self.dropout(x)
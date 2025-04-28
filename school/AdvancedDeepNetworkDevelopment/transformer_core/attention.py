import torch

def ScaledDotProductAttention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None) -> torch.Tensor:
    """Compute scaled dot-product attention

        Args:
            - query: matrix of shape (batch_size, seq_len, d_k)
            - key: matrix of shape (batch_size, seq_len, d_k)
            - value: matrix of shape (batch_size, seq_len, d_v)
            - mask(Optional): mask to exclude tokens
        Output:
            - context matrix after attention applied
    """
    attention_score = torch.matmul(query, torch.transpose(key, -2, -1))
    scaled_attention_score = attention_score / torch.sqrt(torch.tensor(query.shape[-1]))

    if mask is not None:
        scaled_attention_score = scaled_attention_score.masked_fill(mask == 0, -1e10)
    
    normalized_scores = torch.softmax(scaled_attention_score, dim=-1)
    context_matrix = torch.matmul(normalized_scores, value)
    return context_matrix


import torch
import pytest

from transformer_core.attention import ScaledDotProductAttention

def test_scaled_dot_product_attention_output_shape():
    """
    Tests if the output tensor has the expected shape (batch_size, seq_len_q, d_v).
    """
    batch_size = 4
    seq_len_q = 10
    seq_len_k = 12
    d_k = 32
    d_v = 64

    query = torch.randn(batch_size, seq_len_q, d_k)
    key = torch.randn(batch_size, seq_len_k, d_k)
    value = torch.randn(batch_size, seq_len_k, d_v)
    mask = torch.randint(0, 2, (batch_size, 1, seq_len_k)) 

    output = ScaledDotProductAttention(query, key, value, mask=mask)

    expected_shape = (batch_size, seq_len_q, d_v)
    assert output.shape == expected_shape, \
        f"Output shape mismatch: Expected {expected_shape}, got {output.shape}"

def test_scaled_dot_product_attention_padding():
    """
    Tests whether the masking does mask out not necessary values.
    """

    batch_size = 4
    seq_len_q = 10
    seq_len_k = 12
    d_k = 32
    d_v = 64

    query = torch.randn(batch_size, seq_len_q, d_k)
    key = torch.randn(batch_size, seq_len_k, d_k)
    value_original = torch.randn(batch_size, seq_len_k, d_v)

    mask_indices = [2, 4]
    mask = torch.ones(batch_size, 1, seq_len_k, dtype=torch.long)
    for idx in mask_indices:
        if 0 <= idx < seq_len_k:
            mask[:, :, idx] = 0

    output_original = ScaledDotProductAttention(query, key, value_original, mask=mask)

    value_modified = value_original.clone()
    modification_value = 99999
    for idx in mask_indices:
         if 0 <= idx < seq_len_k:
            value_modified[:, idx, :] = modification_value

    unmasked_indices = [i for i in range(seq_len_k) if i not in mask_indices]
    assert torch.equal(value_original[:, unmasked_indices, :], value_modified[:, unmasked_indices, :])
    if mask_indices:
        assert not torch.equal(value_original[:, mask_indices, :], value_modified[:, mask_indices, :])

    output_modified = ScaledDotProductAttention(query, key, value_modified, mask=mask)
    assert torch.allclose(output_original, output_modified, atol=1e-6), \
        "Output changed significantly despite modifications only occurring at masked value positions. Masking might not be working."
    
def test_scaled_dot_product_attention_batching():
    """
    Tests if processing items in a batch gives the same result as
    processing them individually and concatenating.
    """
    batch_size = 3
    seq_len_q = 6
    seq_len_k = 7
    d_k = 16
    d_v = 20

    query = torch.randn(batch_size, seq_len_q, d_k)
    key = torch.randn(batch_size, seq_len_k, d_k)
    value = torch.randn(batch_size, seq_len_k, d_v)
    mask = torch.randint(0, 2, (batch_size, 1, seq_len_k)) 

    output_batch = ScaledDotProductAttention(query, key, value, mask=mask)

    outputs_individual = []
    for i in range(batch_size):
        q_i = query[i:i+1]
        k_i = key[i:i+1]
        v_i = value[i:i+1]
        mask_i = mask[i:i+1] if mask is not None else None

        output_i = ScaledDotProductAttention(q_i, k_i, v_i, mask=mask_i)
        outputs_individual.append(output_i)

    output_concat = torch.cat(outputs_individual, dim=0)

    assert torch.allclose(output_batch, output_concat, atol=1e-6), \
        "Batch processing differs from individual processing."
    


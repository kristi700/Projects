import torch
import pytest

from transformer_core.positional_encoding import PositionalEncoding

def test_positional_encoding_output_shape():
    """
    Tests if the output tensor has the expected shape (batch_size, seq_len, d_model).
    """
    batch_size = 4
    seq_len = 64
    d_model = 32
    positional_encoding = PositionalEncoding(d_model)

    embedding = torch.rand([batch_size, seq_len, d_model])

    positinal_encoded_embedding = positional_encoding(embedding)
    expected_shape = (batch_size, seq_len, d_model)
    assert positinal_encoded_embedding.shape == expected_shape, \
        f"Output shape mismatch: Expected {expected_shape}, got {positinal_encoded_embedding.shape}" 
    
def test_positional_encoding_slicing():
    """
    Tests that the forward pass uses the correct slice of the pe matrix
    corresponding to the input sequence length.
    """
    batch_size = 4
    seq_len = 64
    d_model = 32
    max_len = 128

    positional_encoding = PositionalEncoding(d_model, dropout=0.0, max_length=max_len)
    embedding = torch.zeros(batch_size, seq_len, d_model)

    output = positional_encoding(embedding)
    expected_pe_slice = positional_encoding.pe[:, :seq_len, :]

    assert torch.allclose(output, expected_pe_slice, atol=1e-7), \
        "Output should equal the PE slice when input is zeros and dropout is off."

def test_positional_encoding_values_differ_by_position():
    """
    Tests that PE vectors for different positions are actually different.
    """
    batch_size = 1
    seq_len = 10
    d_model = 32
    max_len = 20

    positional_encoding = PositionalEncoding(d_model, dropout=0.0, max_length=max_len)

    embedding = torch.zeros(batch_size, seq_len, d_model)
    output = positional_encoding(embedding)

    pe_pos0 = output[:, 0, :]
    pe_pos1 = output[:, 1, :]
    pe_pos2 = output[:, 2, :]

    assert not torch.equal(pe_pos0, pe_pos1), "PE vectors for position 0 and 1 should not be identical."
    assert not torch.equal(pe_pos1, pe_pos2), "PE vectors for position 1 and 2 should not be identical."

def test_positional_encoding_value_at_pos0():
    """
    Tests the specific sin/cos pattern at position 0.
    PE(0, 2i) = sin(0 / ...) = 0
    PE(0, 2i+1) = cos(0 / ...) = 1
    """
    d_model = 128
    max_len = 128

    positional_encoding = PositionalEncoding(d_model, dropout=0.0, max_length=max_len)
    pe_matrix = positional_encoding.pe.squeeze(0)

    pe_pos0 = pe_matrix[0, :]

    assert torch.allclose(pe_pos0[0::2], torch.zeros(d_model // 2), atol=1e-7), \
        "Even indices at position 0 should be sin(0) = 0."

    assert torch.allclose(pe_pos0[1::2], torch.ones(d_model // 2), atol=1e-7), \
        "Odd indices at position 0 should be cos(0) = 1."

def test_max_length_assert():
    """
    Tests whether asserting max_length works as intended.
    """
    batch_size = 4
    seq_len = 64
    d_model = 32
    max_len = 32

    positional_encoding = PositionalEncoding(d_model, dropout=0.0, max_length=max_len)
    embedding = torch.zeros(batch_size, seq_len, d_model)

    with pytest.raises(AssertionError):
        _ = positional_encoding(embedding)
import torch
import pytest

from transformer_core.decoder import Decoder, DecoderBlock

@pytest.fixture
def params():
    return {
        "batch_size": 4,
        "tgt_seq_len": 10,
        "src_seq_len": 12,
        "d_model": 64, 
        "num_heads": 8,
        "d_ff": 128,
        "num_blocks": 3,
        "dropout": 0.0 
    }

@pytest.fixture
def target_input(params):
    """Input to the decoder (e.g., target embeddings + positional encodings)."""
    return torch.rand(params["batch_size"], params["tgt_seq_len"], params["d_model"])

@pytest.fixture
def encoder_output_data(params):
    """Output from the encoder stack."""
    return torch.rand(params["batch_size"], params["src_seq_len"], params["d_model"])

@pytest.fixture
def target_mask(params):
    """Example look-ahead mask for self-attention."""
    seq_len = params["tgt_seq_len"]
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    return mask

@pytest.fixture
def source_mask(params):
    """Example padding mask for the source sequence (encoder output)."""
    mask = torch.ones(params["batch_size"], params["src_seq_len"], dtype=torch.bool)
    if params["src_seq_len"] > 2:
        mask[0, -2:] = False
    return mask

def test_decoder_block_forward_shape(params, target_input, encoder_output_data, target_mask, source_mask):
    block = DecoderBlock(d_model=params["d_model"], num_heads=params["num_heads"])
    output = block(target_input, encoder_output_data, look_ahead_mask=target_mask, padding_mask=source_mask)
    assert output.shape == target_input.shape, f"Expected shape {target_input.shape}, but got {output.shape}"
    assert output.dtype == target_input.dtype

def test_decoder_block_forward_shape_no_masks(params, target_input, encoder_output_data):
    block = DecoderBlock(d_model=params["d_model"], num_heads=params["num_heads"])
    output = block(target_input, encoder_output_data, look_ahead_mask=None, padding_mask=None)
    assert output.shape == target_input.shape, f"Expected shape {target_input.shape}, but got {output.shape}"

def test_decoder_block_values_change(params, target_input, encoder_output_data, target_mask, source_mask):
    block = DecoderBlock(d_model=params["d_model"], num_heads=params["num_heads"])
    block.eval() 
    input_clone = target_input.clone()
    output = block(target_input, encoder_output_data, look_ahead_mask=target_mask, padding_mask=source_mask)
    assert not torch.equal(output, input_clone), "Output should be different from input"
    assert torch.equal(target_input, input_clone), "Input tensor should not be modified in-place"

def test_decoder_forward_shape(params, target_input, encoder_output_data, target_mask, source_mask):
    decoder = Decoder(num_blocks=params["num_blocks"], d_model=params["d_model"],
                      num_heads=params["num_heads"])
    output = decoder(target_input, encoder_output_data, look_ahead_mask=target_mask, padding_mask=source_mask)
    assert output.shape == target_input.shape, f"Expected shape {target_input.shape}, but got {output.shape}"
    assert output.dtype == target_input.dtype

def test_decoder_forward_shape_no_masks(params, target_input, encoder_output_data):
    decoder = Decoder(num_blocks=params["num_blocks"], d_model=params["d_model"],
                      num_heads=params["num_heads"])
    output = decoder(target_input, encoder_output_data, look_ahead_mask=None, padding_mask=None)
    assert output.shape == target_input.shape, f"Expected shape {target_input.shape}, but got {output.shape}"

def test_decoder_values_change(params, target_input, encoder_output_data, target_mask, source_mask):
    decoder = Decoder(num_blocks=params["num_blocks"], d_model=params["d_model"],num_heads=params["num_heads"])
    decoder.eval()
    input_clone = target_input.clone()
    output = decoder(target_input, encoder_output_data, look_ahead_mask=target_mask, padding_mask=source_mask)
    assert not torch.equal(output, input_clone), "Output should be different from input after multiple blocks"
    assert torch.equal(target_input, input_clone), "Input tensor should not be modified in-place"

@pytest.mark.parametrize("num_blocks", [1, 5])
def test_decoder_different_num_blocks(params, target_input, encoder_output_data, num_blocks):
    decoder = Decoder(num_blocks=num_blocks, d_model=params["d_model"],
                      num_heads=params["num_heads"])
    output = decoder(target_input, encoder_output_data, look_ahead_mask=None, padding_mask=None)
    assert output.shape == target_input.shape
    assert len(decoder.decoder_blocks) == num_blocks
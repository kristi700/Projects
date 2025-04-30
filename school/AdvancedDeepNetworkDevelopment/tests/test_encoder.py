import torch
import pytest

from transformer_core.encoder import Encoder, EncoderBlock

@pytest.fixture
def params():
    return {
        "batch_size": 4,
        "seq_len": 10,
        "d_model": 512,
        "num_heads": 8,
        "num_blocks": 3
    }

@pytest.fixture
def input_tensor(params):
    return torch.rand(params["batch_size"], params["seq_len"], params["d_model"])

@pytest.fixture
def src_mask(params):
    mask = torch.ones(params["batch_size"], params["seq_len"], dtype=torch.bool)
    if params["seq_len"] > 2:
       mask[0, -2:] = 0
       mask[1, -1:] = 0
    return mask 

def test_encoder_block_forward_shape(params, input_tensor):
    block = EncoderBlock(d_model=params["d_model"], num_heads=params["num_heads"])
    output = block(input_tensor, mask=None)
    assert output.shape == input_tensor.shape

def test_encoder_block_forward_shape_with_mask(params, input_tensor, src_mask):
    block = EncoderBlock(d_model=params["d_model"], num_heads=params["num_heads"])
    
    mask_for_mha = src_mask.unsqueeze(1).unsqueeze(1) 
    output = block(input_tensor, mask=mask_for_mha)
    assert output.shape == input_tensor.shape


def test_encoder_forward_shape(params, input_tensor):
    encoder = Encoder(num_blocks=params["num_blocks"], d_model=params["d_model"], num_heads=params["num_heads"])
    output = encoder(input_tensor, mask=None)
    assert output.shape == input_tensor.shape

def test_encoder_forward_shape_with_mask(params, input_tensor, src_mask):
    encoder = Encoder(num_blocks=params["num_blocks"], d_model=params["d_model"], num_heads=params["num_heads"])
    
    mask_for_mha = src_mask.unsqueeze(1).unsqueeze(1) 
    output = encoder(input_tensor, mask=mask_for_mha)
    assert output.shape == input_tensor.shape

@pytest.mark.parametrize("num_blocks", [1, 4])
def test_encoder_different_num_blocks(params, input_tensor, num_blocks):
    encoder = Encoder(num_blocks=num_blocks, d_model=params["d_model"], num_heads=params["num_heads"])
    output = encoder(input_tensor, mask=None)
    assert output.shape == input_tensor.shape
    assert len(encoder.encoder_blocks) == num_blocks
import torch
import pytest

from transformer_core.transformer import Transformer

@pytest.fixture(scope="module") 
def params():
    return {
        "batch_size": 4,
        "src_seq_len": 15,
        "tgt_seq_len": 12,
        "src_vocab_size": 100,
        "tgt_vocab_size": 110,
        "d_model": 64, 
        "num_heads": 8,
        "num_blocks": 3,
    }

@pytest.fixture(scope="module")
def transformer_model(params):
    """Provides an instance of the Transformer model."""
    model = Transformer(
        src_vocab_size=params["src_vocab_size"],
        tgt_vocab_size=params["tgt_vocab_size"],
        num_blocks=params["num_blocks"],
        d_model=params["d_model"],
        num_heads=params["num_heads"],
    )
    model.eval() 
    return model

@pytest.fixture
def sample_src_tokens(params):
    """Provides sample source token IDs."""
    return torch.randint(1, params["src_vocab_size"], (params["batch_size"], params["src_seq_len"]))

@pytest.fixture
def sample_tgt_tokens(params):
    """Provides sample target token IDs (e.g., shifted right)."""
    return torch.randint(1, params["tgt_vocab_size"], (params["batch_size"], params["tgt_seq_len"]))

@pytest.fixture
def sample_padding_mask(params, sample_src_tokens):
    """
    Creates a sample source padding mask (True=Keep, False=Mask).
    Example: Mask last 3 tokens for batch item 0, last 1 for item 1.
    Shape: (B, S)
    """
    mask = torch.ones_like(sample_src_tokens, dtype=torch.bool)
    if params["src_seq_len"] > 3:
        mask[0, -3:] = False
    if params["src_seq_len"] > 1:
        mask[1, -1:] = False
    return mask

@pytest.fixture
def sample_look_ahead_mask(params):
    """
    Creates a combined look-ahead and target padding mask (True=Keep, False=Mask).
    For simplicity, just using look-ahead here.
    Shape: (T, T)
    """
    seq_len = params["tgt_seq_len"]
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    return mask

def test_transformer_forward_shape(transformer_model, params, sample_src_tokens, sample_tgt_tokens, sample_look_ahead_mask, sample_padding_mask):
    """Tests the output shape of the forward pass."""
    output = transformer_model(
        x=sample_src_tokens,
        y=sample_tgt_tokens,
        look_ahead_mask=sample_look_ahead_mask,
        padding_mask=sample_padding_mask
    )

    expected_shape = (params["batch_size"], params["tgt_seq_len"], params["tgt_vocab_size"])
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, but got {output.shape}"
    assert output.dtype == torch.float32 

@pytest.mark.parametrize("batch_size", [1, 5]) 
def test_transformer_forward_batch_sizes(params, batch_size):
    """Tests forward pass with different batch sizes."""
    
    current_params = params.copy()
    current_params["batch_size"] = batch_size

    model = Transformer(
        src_vocab_size=current_params["src_vocab_size"],
        tgt_vocab_size=current_params["tgt_vocab_size"],
        num_blocks=current_params["num_blocks"],
        d_model=current_params["d_model"],
        num_heads=current_params["num_heads"]
    )
    model.eval()

    x = torch.randint(1, current_params["src_vocab_size"], (current_params["batch_size"], current_params["src_seq_len"]), dtype=torch.long)
    y = torch.randint(1, current_params["tgt_vocab_size"], (current_params["batch_size"], current_params["tgt_seq_len"]), dtype=torch.long)
    padding_mask = torch.ones_like(x, dtype=torch.bool) 
    look_ahead_mask = torch.tril(torch.ones(current_params["tgt_seq_len"], current_params["tgt_seq_len"], dtype=torch.bool))

    output = model(x, y, look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)

    expected_shape = (current_params["batch_size"], current_params["tgt_seq_len"], current_params["tgt_vocab_size"])
    assert output.shape == expected_shape

def test_transformer_forward_value_change(transformer_model, sample_src_tokens, sample_tgt_tokens, sample_look_ahead_mask, sample_padding_mask):
    """Tests if the forward pass actually changes the values (not identity)."""
    src_clone = sample_src_tokens.clone()
    tgt_clone = sample_tgt_tokens.clone()

    output = transformer_model(
        x=sample_src_tokens,
        y=sample_tgt_tokens,
        look_ahead_mask=sample_look_ahead_mask,
        padding_mask=sample_padding_mask
    )
    
    assert torch.equal(src_clone, sample_src_tokens)
    assert torch.equal(tgt_clone, sample_tgt_tokens)
    assert not torch.any(output == sample_tgt_tokens.unsqueeze(-1).float()), "Logits matched input token IDs"

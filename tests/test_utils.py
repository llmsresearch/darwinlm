import pytest
import torch
from darwinlm.utils.metrics import compute_kl_divergence, compute_loss
from darwinlm.utils.config import load_config
from darwinlm.utils.checkpointing import CheckpointManager

def test_kl_divergence():
    p = torch.randn(2, 3)
    q = torch.randn(2, 3)
    kl_div = compute_kl_divergence(p, q)
    assert kl_div.ndim == 0  # scalar output
    assert kl_div >= 0  # KL divergence is non-negative

def test_config_loading(tmp_path):
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        f.write("model:\n  name: test\n")
    
    config = load_config(config_path)
    assert config.model.name == "test"

def test_checkpointing(tmp_path):
    config = {"output_dir": str(tmp_path)}
    manager = CheckpointManager(config)
    
    # Test checkpoint saving
    model = torch.nn.Linear(10, 10)
    manager.save_checkpoint(model, 1, 0.5, [5] * 10)
    
    assert (tmp_path / "checkpoints" / "gen_1.pt").exists() 
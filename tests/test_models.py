import pytest
from darwinlm.models.model_adapter import ModelAdapter

@pytest.fixture
def llama_config():
    return {
        "model": {
            "name": "llama2-7b",
            "pretrained_path": "meta-llama/Llama-2-7b-hf",
            "dtype": "bfloat16",
            "device": "cuda"
        }
    }

def test_model_adapter_initialization(llama_config):
    adapter = ModelAdapter(llama_config)
    assert adapter.model_family == "llama"
    
def test_unsupported_model():
    config = {"model": {"name": "unsupported-model"}}
    with pytest.raises(ValueError):
        ModelAdapter(config)

def test_attention_config(llama_config):
    adapter = ModelAdapter(llama_config)
    config = adapter.get_attention_config()
    
    assert "num_heads" in config
    assert "head_dim" in config
    assert "is_gqa" in config 
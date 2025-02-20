from typing import Dict, Any, Union, Optional
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from dataclasses import dataclass

@dataclass
class ModelArchitectureConfig:
    """Configuration for different model architectures"""
    num_attention_heads: int
    head_dim: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    is_gqa: bool = False
    attention_head_size: Optional[int] = None
    num_key_value_heads: Optional[int] = None

class ModelAdapter:
    """Adapter class to handle different LLM architectures"""
    
    SUPPORTED_MODELS = {
        "llama": {
            "attention_pattern": "standard",
            "has_gqa": False
        },
        "qwen": {
            "attention_pattern": "grouped",
            "has_gqa": True
        },
        "mistral": {
            "attention_pattern": "grouped",
            "has_gqa": True
        },
        "falcon": {
            "attention_pattern": "standard",
            "has_gqa": False
        },
        "mpt": {
            "attention_pattern": "standard",
            "has_gqa": False
        }
    }
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config["model"]["name"].lower()
        self.model = None
        self.model_config = None
        self._validate_model_type()
        
    def _validate_model_type(self):
        """Validate if model type is supported"""
        model_family = next(
            (k for k in self.SUPPORTED_MODELS.keys() if k in self.model_name), 
            None
        )
        if not model_family:
            raise ValueError(
                f"Unsupported model type: {self.model_name}. "
                f"Supported models: {list(self.SUPPORTED_MODELS.keys())}"
            )
        self.model_family = model_family
        
    def load_model(self) -> nn.Module:
        """Load pretrained model with appropriate configuration"""
        self.model_config = AutoConfig.from_pretrained(
            self.config["model"]["pretrained_path"]
        )
        
        # Handle model-specific loading configurations
        load_kwargs = {
            "config": self.model_config,
            "torch_dtype": getattr(torch, self.config["model"]["dtype"]),
            "device_map": self.config["model"]["device"]
        }
        
        # Add model-specific loading args
        if self.model_family in ["llama", "mistral"]:
            load_kwargs["use_flash_attention_2"] = True
        elif self.model_family == "qwen":
            load_kwargs["use_flash_attention"] = True
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["model"]["pretrained_path"],
            **load_kwargs
        )
        return self.model
    
    def get_architecture_config(self) -> ModelArchitectureConfig:
        """Get complete model architecture configuration"""
        config = self.model_config
        
        base_config = {
            "num_attention_heads": config.num_attention_heads,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_hidden_layers": config.num_hidden_layers,
            "is_gqa": self.SUPPORTED_MODELS[self.model_family]["has_gqa"]
        }
        
        # Handle different attention patterns
        if self.SUPPORTED_MODELS[self.model_family]["attention_pattern"] == "grouped":
            base_config.update({
                "head_dim": config.hidden_size // config.num_attention_heads,
                "num_key_value_heads": getattr(config, "num_key_value_heads", 
                                             config.num_attention_heads),
                "attention_head_size": getattr(config, "attention_head_size", 
                                             config.hidden_size // config.num_attention_heads)
            })
        else:
            base_config.update({
                "head_dim": config.hidden_size // config.num_attention_heads
            })
            
        return ModelArchitectureConfig(**base_config)
    
    def get_attention_config(self) -> Dict[str, int]:
        """Get attention-related configuration for the model"""
        arch_config = self.get_architecture_config()
        
        config = {
            "num_heads": arch_config.num_attention_heads,
            "head_dim": arch_config.head_dim,
            "is_gqa": arch_config.is_gqa
        }
        
        if arch_config.is_gqa:
            config.update({
                "num_kv_heads": arch_config.num_key_value_heads,
                "attention_head_size": arch_config.attention_head_size
            })
            
        return config
    
    def get_mlp_config(self) -> Dict[str, int]:
        """Get MLP-related configuration for the model"""
        arch_config = self.get_architecture_config()
        return {
            "hidden_size": arch_config.hidden_size,
            "intermediate_size": arch_config.intermediate_size
        }
        
    def get_layer_structure(self) -> Dict[str, Any]:
        """Get layer structure information for pruning"""
        arch_config = self.get_architecture_config()
        
        return {
            "num_layers": arch_config.num_hidden_layers,
            "attention": self.get_attention_config(),
            "mlp": self.get_mlp_config(),
            "attention_pattern": self.SUPPORTED_MODELS[self.model_family]["attention_pattern"]
        } 
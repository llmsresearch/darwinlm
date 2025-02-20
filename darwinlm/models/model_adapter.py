from typing import Dict, Any, Union, Optional
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig, PreTrainedModel
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
    """Handles model loading and adaptation"""
    
    def __init__(self,
                 model_name_or_path: str,
                 config: Dict[str, Any]):
        self.model_name = model_name_or_path
        self.config = config
        
        # Detect if model uses GQA
        self.is_gqa = self._detect_gqa()
        self.config["model"]["is_gqa"] = self.is_gqa
        
    def _detect_gqa(self) -> bool:
        """Detect if model uses Group Query Attention"""
        model_config = AutoConfig.from_pretrained(self.model_name)
        
        # Check for GQA indicators in different model types
        if hasattr(model_config, "num_key_value_heads"):
            return model_config.num_key_value_heads != model_config.num_attention_heads
        elif hasattr(model_config, "num_kv_heads"):
            return model_config.num_kv_heads != model_config.num_attention_heads
            
        return False
        
    def load_model(self) -> PreTrainedModel:
        """Load and prepare model
        
        Returns:
            PreTrainedModel: Loaded model
        """
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Move to device if specified
        if "device" in self.config["hardware"]:
            device = torch.device(self.config["hardware"]["device"])
            model = model.to(device)
            
        return model
        
    def save_model(self,
                  model: PreTrainedModel,
                  output_dir: str,
                  save_name: Optional[str] = None):
        """Save model checkpoint
        
        Args:
            model: Model to save
            output_dir: Output directory
            save_name: Name for saved model (optional)
        """
        if save_name:
            output_dir = f"{output_dir}/{save_name}"
            
        model.save_pretrained(output_dir)
        
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration
        
        Returns:
            Dict[str, Any]: Model configuration
        """
        model_config = AutoConfig.from_pretrained(self.model_name)
        
        return {
            "num_attention_heads": model_config.num_attention_heads,
            "num_hidden_layers": model_config.num_hidden_layers,
            "hidden_size": model_config.hidden_size,
            "intermediate_size": model_config.intermediate_size,
            "is_gqa": self.is_gqa
        }
    
    def get_architecture_config(self) -> ModelArchitectureConfig:
        """Get complete model architecture configuration"""
        config = AutoConfig.from_pretrained(self.model_name)
        
        base_config = {
            "num_attention_heads": config.num_attention_heads,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_hidden_layers": config.num_hidden_layers,
            "is_gqa": self.is_gqa
        }
        
        # Handle different attention patterns
        if hasattr(config, "num_key_value_heads"):
            base_config.update({
                "head_dim": config.hidden_size // config.num_attention_heads,
                "num_key_value_heads": config.num_key_value_heads,
                "attention_head_size": config.hidden_size // config.num_attention_heads
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
            "attention_pattern": "standard"  # Assuming standard attention pattern
        } 
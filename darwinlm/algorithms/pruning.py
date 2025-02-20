import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, List
from dataclasses import dataclass
from ..utils.metrics import compute_hessian

@dataclass
class PruningMask:
    """Represents pruning decisions for a layer"""
    attention_mask: torch.Tensor  # Mask for attention heads
    mlp_mask: torch.Tensor       # Mask for MLP dimensions
    weight_mask: torch.Tensor    # Fine-grained weight mask
    layer_idx: int
    sparsity_level: float

class SecondOrderPruning:
    """Second-order structured pruning using OBS methodology"""
    
    def __init__(self, 
                 model: nn.Module, 
                 config: Dict[str, Any],
                 calibration_data: torch.Tensor):
        self.model = model
        self.config = config
        self.calibration_data = calibration_data
        self.device = config["hardware"]["device"]
        self.mlp_granularity = 32  # As per paper
        self.is_gqa = config.get("model", {}).get("is_gqa", False)
        
    def compute_hessian(self, layer_inputs: torch.Tensor) -> torch.Tensor:
        """Compute Hessian matrix H = XX^T for l2 minimization"""
        return torch.matmul(layer_inputs, layer_inputs.transpose(-1, -2))
    
    def compute_optimal_mask(self, 
                           weights: torch.Tensor, 
                           hessian: torch.Tensor, 
                           num_dims: int,
                           is_mlp: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute optimal structured mask and weight updates using OBS"""
        # Adjust num_dims to be multiple of granularity for MLP
        if is_mlp:
            num_dims = (num_dims // self.mlp_granularity) * self.mlp_granularity
            
        # Get inverse of Hessian submatrix
        h_inv = torch.inverse(hessian)
        
        # Compute importance scores for each dimension
        scores = []
        for i in range(weights.size(0)):
            w_i = weights[i]
            h_inv_i = h_inv[i:i+1, i:i+1]
            score = torch.matmul(torch.matmul(w_i, h_inv_i), w_i.t())
            scores.append(score.item())
            
        # Get mask by keeping top-k dimensions
        sorted_idx = torch.argsort(torch.tensor(scores), descending=True)
        mask = torch.zeros_like(weights)
        mask[sorted_idx[:num_dims]] = 1
        
        # Compute weight updates
        w_pruned = weights * mask
        delta = -torch.matmul(torch.matmul(w_pruned, h_inv), hessian)
        
        return mask, delta
        
    def prune_layer(self, 
                    layer: nn.Module, 
                    sparsity: float,
                    layer_idx: int) -> Tuple[nn.Module, PruningMask]:
        """Prune a single layer using second-order information"""
        # Get layer inputs by forward pass
        with torch.no_grad():
            inputs = []
            def hook(module, inp, out):
                inputs.append(inp[0])
            handle = layer.register_forward_hook(hook)
            _ = self.model(self.calibration_data)
            handle.remove()
            layer_inputs = inputs[0]
        
        # Compute Hessian
        H = self.compute_hessian(layer_inputs)
        
        # Get number of dimensions to keep
        num_dims = int(layer.weight.size(0) * (1 - sparsity))
        
        is_attention = "attention" in layer._get_name().lower()
        is_mlp = "mlp" in layer._get_name().lower()
        
        # Get mask and updates
        mask, delta = self.compute_optimal_mask(
            layer.weight.data, 
            H, 
            num_dims,
            is_mlp=is_mlp
        )
        
        # Create pruning mask object
        pruning_mask = PruningMask(
            attention_mask=mask if is_attention else None,
            mlp_mask=mask if is_mlp else None,
            weight_mask=mask,
            layer_idx=layer_idx,
            sparsity_level=sparsity
        )
        
        # Apply pruning
        layer.weight.data = layer.weight.data * mask + delta
        
        # Handle GQA for attention layers
        if is_attention and self.is_gqa:
            # Don't prune K,V matrices for GQA
            if hasattr(layer, 'k_proj'):
                layer.k_proj.weight.data = layer.k_proj.weight.data
            if hasattr(layer, 'v_proj'):
                layer.v_proj.weight.data = layer.v_proj.weight.data
            # Only prune Q matrix
            if hasattr(layer, 'q_proj'):
                layer.q_proj.weight.data = layer.q_proj.weight.data * mask + delta
        
        return layer, pruning_mask 
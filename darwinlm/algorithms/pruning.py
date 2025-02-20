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
        
    def compute_hessian(self, layer_inputs: torch.Tensor) -> torch.Tensor:
        """Compute Hessian matrix H = XX^T for l2 minimization"""
        return torch.matmul(layer_inputs, layer_inputs.transpose(-1, -2))
    
    def compute_optimal_mask(self, 
                           weights: torch.Tensor, 
                           hessian: torch.Tensor, 
                           num_dims: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute optimal structured mask and weight updates using OBS"""
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
        
        # Get mask and updates
        mask, delta = self.compute_optimal_mask(layer.weight.data, H, num_dims)
        
        # Create pruning mask object
        pruning_mask = PruningMask(
            attention_mask=mask if "attention" in layer._get_name().lower() else None,
            mlp_mask=mask if "mlp" in layer._get_name().lower() else None,
            weight_mask=mask,
            layer_idx=layer_idx,
            sparsity_level=sparsity
        )
        
        # Apply pruning
        layer.weight.data = layer.weight.data * mask + delta
        
        return layer, pruning_mask 
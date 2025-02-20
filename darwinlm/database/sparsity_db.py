import torch
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class SparsityLevel:
    attention_heads: int
    mlp_dims: int
    layer_idx: int
    pruned_weights: torch.Tensor

class SparsityDatabase:
    """Database to store pruned layer weights at different sparsity levels"""
    
    def __init__(self, num_levels: int):
        self.num_levels = num_levels
        self.db: Dict[int, List[SparsityLevel]] = {}
        
    def compute_level_sparsity(self, level: int, 
                              num_heads: int, 
                              intermediate_size: int,
                              granularity: int = 32) -> Tuple[int, int]:
        """Compute number of heads/dims to prune for a given level"""
        num_pruned_heads = round(level * num_heads / self.num_levels)
        num_pruned_dims = granularity * round(level * intermediate_size / 
                                            (self.num_levels * granularity))
        return num_pruned_heads, num_pruned_dims
        
    def add_level(self, level: int, layer_idx: int, 
                  pruned_weights: torch.Tensor,
                  num_heads: int, num_dims: int):
        """Add pruned weights for a sparsity level"""
        if level not in self.db:
            self.db[level] = []
            
        self.db[level].append(
            SparsityLevel(
                attention_heads=num_heads,
                mlp_dims=num_dims,
                layer_idx=layer_idx,
                pruned_weights=pruned_weights
            )
        )
        
    def get_level(self, level: int, layer_idx: int) -> SparsityLevel:
        """Retrieve pruned weights for a specific level and layer"""
        levels = self.db.get(level, [])
        for l in levels:
            if l.layer_idx == layer_idx:
                return l
        raise KeyError(f"No data found for level {level}, layer {layer_idx}") 
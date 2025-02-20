import torch
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class SparsityLevel:
    """Represents a sparsity level for a module"""
    attention_heads: int  # Number of pruned heads
    mlp_dims: int        # Number of pruned MLP dimensions
    sparsity: float      # Actual sparsity achieved

class SparsityDatabase:
    """Database of pre-computed sparsity levels following paper equations 5-6"""
    
    def __init__(self,
                 num_levels: int,
                 num_heads: int,
                 intermediate_size: int,
                 mlp_granularity: int = 32):
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.mlp_granularity = mlp_granularity
        self.levels: Dict[int, SparsityLevel] = {}
        
        # Generate sparsity levels
        self._generate_levels()
        
    def _generate_levels(self):
        """Generate sparsity levels according to equations 5-6"""
        for i in range(self.num_levels + 1):
            # Equation 5: Number of pruned attention heads
            pruned_heads = round(i * self.num_heads / self.num_levels)
            
            # Equation 6: Number of pruned MLP dimensions
            pruned_mlp = self.mlp_granularity * round(
                i * self.intermediate_size / (self.num_levels * self.mlp_granularity)
            )
            
            # Calculate actual sparsity achieved
            total_params = self.num_heads + self.intermediate_size
            pruned_params = pruned_heads + pruned_mlp
            sparsity = pruned_params / total_params
            
            self.levels[i] = SparsityLevel(
                attention_heads=pruned_heads,
                mlp_dims=pruned_mlp,
                sparsity=sparsity
            )
    
    def get_level(self, level: int) -> Optional[SparsityLevel]:
        """Get sparsity configuration for a given level"""
        return self.levels.get(level)
    
    def get_closest_level(self, target_sparsity: float) -> int:
        """Find level that gives closest sparsity to target"""
        min_diff = float('inf')
        closest_level = 0
        
        for level, config in self.levels.items():
            diff = abs(config.sparsity - target_sparsity)
            if diff < min_diff:
                min_diff = diff
                closest_level = level
                
        return closest_level
    
    def get_uniform_levels(self, target_sparsity: float, num_layers: int) -> List[int]:
        """Get uniform level allocation for all layers"""
        level = self.get_closest_level(target_sparsity)
        return [level] * num_layers 
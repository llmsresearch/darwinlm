import torch
import os
from typing import Dict, Any, List

class CheckpointManager:
    """Manage model checkpoints during evolution"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.checkpoint_dir = os.path.join(config["output_dir"], "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, 
                       model: torch.nn.Module,
                       generation: int,
                       best_fitness: float,
                       sparsity_levels: List[int]):
        """Save evolution checkpoint"""
        checkpoint = {
            "model_state": model.state_dict(),
            "generation": generation,
            "best_fitness": best_fitness,
            "sparsity_levels": sparsity_levels
        }
        
        path = os.path.join(self.checkpoint_dir, f"gen_{generation}.pt")
        torch.save(checkpoint, path) 
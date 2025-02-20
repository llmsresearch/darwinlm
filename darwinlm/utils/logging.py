import wandb
import logging
from typing import Dict, Any, List

class ExperimentLogger:
    """Logger for tracking experiments"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_logging()
        
        if config.get("wandb", {}).get("enabled", False):
            wandb.init(
                project=config["wandb"]["project"],
                name=config["wandb"]["run_name"],
                config=config
            )
    
    def log_evolution_step(self, 
                          generation: int, 
                          best_fitness: float,
                          sparsity_levels: List[int],
                          model_size: int):
        """Log evolution progress"""
        metrics = {
            "generation": generation,
            "best_fitness": best_fitness,
            "model_size_mb": model_size / (1024 * 1024),
            "avg_sparsity": sum(sparsity_levels) / len(sparsity_levels)
        }
        
        if wandb.run is not None:
            wandb.log(metrics)
        
        logging.info(f"Generation {generation}: Best Fitness = {best_fitness:.4f}") 
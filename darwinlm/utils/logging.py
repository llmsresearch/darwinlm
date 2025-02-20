import wandb
import logging
from typing import Dict, Any, List
import os
from pathlib import Path
from typing import Optional

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

def setup_logging(log_dir: str, 
                 log_file: Optional[str] = "darwinlm.log",
                 level: int = logging.INFO):
    """Setup logging configuration
    
    Args:
        log_dir: Directory to save log files
        log_file: Name of log file (optional)
        level: Logging level
    """
    # Create log directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Configure logging format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup file handler
    if log_file:
        file_handler = logging.FileHandler(
            os.path.join(log_dir, log_file)
        )
        file_handler.setFormatter(formatter)
        
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    if log_file:
        root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Log basic info
    logger = logging.getLogger(__name__)
    logger.info("Logging setup complete")
    logger.info(f"Log directory: {log_dir}")
    if log_file:
        logger.info(f"Log file: {log_file}")
        
def log_config(config: dict, logger: logging.Logger):
    """Log configuration parameters
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
    """
    logger.info("Configuration:")
    for section, params in config.items():
        logger.info(f"\n[{section}]")
        for key, value in params.items():
            logger.info(f"{key}: {value}")
            
def log_metrics(metrics: dict, step: int, logger: logging.Logger):
    """Log training/evaluation metrics
    
    Args:
        metrics: Dictionary of metrics
        step: Current step number
        logger: Logger instance
    """
    logger.info(f"\nStep {step} metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}") 
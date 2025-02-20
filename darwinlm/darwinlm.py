import os
import yaml
import torch
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from .models.model_adapter import ModelAdapter
from .algorithms.pruning import SecondOrderPruning
from .evolution.search import EvolutionarySearch
from .training.trainer import Trainer
from .database.sparsity_db import SparsityDatabase
from .data.data_manager import DataManager
from .utils.logging import setup_logging

class DarwinLM:
    """Main class for DarwinLM model compression
    
    This class implements the complete DarwinLM pipeline as described in the paper:
    "DarwinLM: Evolutionary Structured Pruning of Large Language Models"
    
    The pipeline consists of:
    1. Second-order structured pruning to generate sparsity level database
    2. Training-aware evolutionary search to find optimal sparsity allocation
    3. Post-training to recover performance
    """
    
    def __init__(self, 
                 config_path: str,
                 model_name_or_path: Optional[str] = None):
        # Load and validate configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Setup logging
        setup_logging(self.config["output"]["log_dir"])
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        self._setup_directories()
        
        # Initialize components
        self.model_adapter = ModelAdapter(
            model_name_or_path or self.config.get("model", {}).get("name"),
            self.config
        )
        self.data_manager = DataManager(self.config)
        
    def _setup_directories(self):
        """Create necessary output directories"""
        dirs = [
            self.config["output"]["save_dir"],
            self.config["output"]["log_dir"],
            self.config["output"]["checkpoint_dir"]
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
            
    def compress(self) -> torch.nn.Module:
        """Main compression pipeline
        
        Returns:
            torch.nn.Module: Compressed model after post-training
        """
        self.logger.info("Starting DarwinLM compression pipeline")
        
        # Load model and data
        model = self.model_adapter.load_model()
        train_loader = self.data_manager.get_train_loader()
        eval_loader = self.data_manager.get_eval_loader()
        calibration_loader = self.data_manager.get_calibration_loader()
        
        # Initialize sparsity database
        sparsity_db = SparsityDatabase(
            num_levels=self.config["pruning"]["num_levels"],
            num_heads=model.config.num_attention_heads,
            intermediate_size=model.config.intermediate_size,
            mlp_granularity=self.config["model"]["mlp_granularity"]
        )
        
        # Initialize trainer
        trainer = Trainer(
            dense_model=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            sparsity_db=sparsity_db,
            config=self.config
        )
        
        # Initialize evolutionary search
        search = EvolutionarySearch(
            sparsity_db=sparsity_db,
            trainer=trainer,
            num_generations=self.config["evolution"]["num_generations"],
            offspring_size=self.config["evolution"]["offspring_size"],
            selection_steps=self.config["evolution"]["selection_steps"],
            finetune_tokens=self.config["evolution"]["finetune_tokens"],
            selection_tokens=self.config["evolution"]["selection_tokens"]
        )
        
        # Run evolutionary search
        self.logger.info("Starting evolutionary search")
        best_levels, best_model = search.search(
            model,
            target_sparsity=self.config["pruning"]["target_sparsity"]
        )
        
        # Save best model before post-training
        self._save_checkpoint(best_model, "best_before_posttraining.pt")
        
        # Post-training
        self.logger.info("Starting post-training")
        final_model = trainer.finetune(
            best_model,
            num_tokens=self.config["training"]["max_tokens"],
            batch_size=self.config["training"]["batch_size"]
        )
        
        # Save final model
        self._save_checkpoint(final_model, "final_model.pt")
        
        return final_model
    
    def _save_checkpoint(self, model: torch.nn.Module, filename: str):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(
            self.config["output"]["checkpoint_dir"],
            filename
        )
        torch.save(model.state_dict(), checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                       help="Path to config file")
    parser.add_argument("--model", type=str, default=None,
                       help="Model name or path (optional)")
    args = parser.parse_args()
    
    darwinlm = DarwinLM(args.config, args.model)
    compressed_model = darwinlm.compress() 
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from typing import Dict, Any, Tuple

class DataManager:
    """Handles data loading and processing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model"]["pretrained_path"]
        )
        
    def get_dataloaders(self, 
                        num_samples: int = None) -> Tuple[DataLoader, DataLoader]:
        """Get train and evaluation dataloaders"""
        train_dataset = self._load_dataset(
            self.config["data"]["train"]["path"],
            num_samples
        )
        eval_dataset = self._load_dataset(
            self.config["data"]["eval"]["path"],
            self.config["data"]["eval"]["num_samples"]
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            num_workers=self.config["hardware"]["num_workers"]
        )
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config["data"]["eval"]["batch_size"],
            shuffle=False,
            num_workers=self.config["hardware"]["num_workers"]
        )
        
        return train_loader, eval_loader 
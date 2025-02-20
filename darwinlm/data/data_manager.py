import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Tuple
import json

class FinewebDataset(Dataset):
    """Dataset class for Fineweb-Edu data"""
    
    def __init__(self, data_path: str, tokenizer: AutoTokenizer, 
                 min_score: float, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load and filter data
        with open(data_path, 'r') as f:
            self.data = [
                json.loads(line) for line in f 
                if float(json.loads(line)["score"]) >= min_score
            ]
            
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Tokenize text
        encodings = self.tokenizer(
            item["text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze()
        }

class DataManager:
    """Manager class for handling data loading and processing"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model"]["pretrained_path"]
        )
        
    def get_dataloaders(self, num_samples: int = None) -> Tuple[DataLoader, DataLoader]:
        """Get train and evaluation dataloaders"""
        dataset = FinewebDataset(
            self.config["data"]["train_path"],
            self.tokenizer,
            self.config["data"]["min_score"],
            self.config["data"]["max_length"]
        )
        
        if num_samples:
            train_size = int(0.9 * num_samples)
            eval_size = num_samples - train_size
            train_dataset, eval_dataset = torch.utils.data.random_split(
                dataset, [train_size, eval_size]
            )
        else:
            train_size = int(0.9 * len(dataset))
            eval_size = len(dataset) - train_size
            train_dataset, eval_dataset = torch.utils.data.random_split(
                dataset, [train_size, eval_size]
            )
            
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True
        )
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False
        )
        
        return train_loader, eval_loader 
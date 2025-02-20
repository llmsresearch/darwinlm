import os
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer, AutoTokenizer
from datasets import load_dataset

class TextDataset(Dataset):
    """Dataset for text data"""
    
    def __init__(self,
                 data_path: str,
                 tokenizer: PreTrainedTokenizer,
                 max_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load dataset
        self.dataset = load_dataset(
            'text',
            data_files=data_path,
            split='train'
        )
        
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.dataset[idx]['text']
        
        # Tokenize
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze()
        }

class DataManager:
    """Handles data loading and preprocessing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.get("model", {}).get("name", "facebook/opt-125m")
        )
        
        # Create datasets
        self.train_dataset = self._create_dataset(
            config["data"]["train_path"]
        )
        self.eval_dataset = self._create_dataset(
            config["data"]["eval_path"]
        )
        self.calibration_dataset = self._create_dataset(
            config["data"]["calibration_path"]
        )
        
    def _create_dataset(self, data_path: str) -> TextDataset:
        """Create dataset from data path"""
        return TextDataset(
            data_path=data_path,
            tokenizer=self.tokenizer,
            max_length=self.config["data"]["max_seq_length"]
        )
        
    def get_train_loader(self, batch_size: Optional[int] = None) -> DataLoader:
        """Get training data loader"""
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size or self.config["training"]["batch_size"],
            shuffle=True,
            num_workers=os.cpu_count()
        )
        
    def get_eval_loader(self, batch_size: Optional[int] = None) -> DataLoader:
        """Get evaluation data loader"""
        return DataLoader(
            self.eval_dataset,
            batch_size=batch_size or self.config["training"]["batch_size"],
            shuffle=False,
            num_workers=os.cpu_count()
        )
        
    def get_calibration_loader(self) -> DataLoader:
        """Get calibration data loader"""
        return DataLoader(
            self.calibration_dataset,
            batch_size=self.config["pruning"]["calibration_size"],
            shuffle=False,
            num_workers=os.cpu_count()
        ) 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from typing import Dict, List

class DarwinTrainer:
    """Trainer class for fine-tuning and evaluating models"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config["model"]["device"])
        
    def finetune(self, 
                 model: nn.Module, 
                 dataloader: DataLoader, 
                 num_tokens: int) -> nn.Module:
        """Finetune model on small dataset"""
        model.train()
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"]
        )
        
        num_steps = num_tokens // self.config["training"]["batch_size"]
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_steps * self.config["training"]["warmup_ratio"]),
            num_training_steps=num_steps
        )
        
        # Training loop
        tokens_seen = 0
        while tokens_seen < num_tokens:
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = model(**batch)
                loss = outputs.loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    self.config["training"]["max_grad_norm"]
                )
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                tokens_seen += batch["input_ids"].numel()
                if tokens_seen >= num_tokens:
                    break
                    
        return model
    
    def evaluate_kl(self, 
                    dense_model: nn.Module, 
                    pruned_model: nn.Module, 
                    eval_dataloader: DataLoader) -> float:
        """Compute KL divergence between dense and pruned model outputs"""
        dense_model.eval()
        pruned_model.eval()
        
        total_kl = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                dense_logits = dense_model(**batch).logits
                pruned_logits = pruned_model(**batch).logits
                
                # Compute KL divergence
                dense_probs = torch.softmax(dense_logits, dim=-1)
                pruned_probs = torch.softmax(pruned_logits, dim=-1)
                
                kl = torch.sum(dense_probs * (torch.log(dense_probs) - 
                                            torch.log(pruned_probs)))
                
                total_kl += kl.item()
                num_batches += 1
                
        return total_kl / num_batches 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from typing import Dict, List, Optional, Any
from ..database.sparsity_db import SparsityDatabase

class Trainer:
    """Handles model training during evolutionary search"""
    
    def __init__(self,
                 dense_model: nn.Module,
                 train_loader: DataLoader,
                 eval_loader: DataLoader,
                 sparsity_db: SparsityDatabase,
                 config: Dict[str, Any]):
        self.dense_model = dense_model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.sparsity_db = sparsity_db
        self.config = config
        self.device = config["hardware"]["device"]
        
        # Training hyperparameters from paper
        self.learning_rate = 1e-5
        self.warmup_steps = 50
        self.weight_decay = 0.0
        
    def get_dense_model(self) -> nn.Module:
        """Return reference dense model"""
        return self.dense_model
        
    def get_eval_batch(self, num_tokens: int) -> torch.Tensor:
        """Get batch of tokens for evaluation"""
        total_tokens = 0
        eval_tokens = []
        
        for batch in self.eval_loader:
            eval_tokens.append(batch)
            total_tokens += batch.numel()
            if total_tokens >= num_tokens:
                break
                
        return torch.cat(eval_tokens, dim=0)[:num_tokens]
        
    def stitch_model(self, 
                    base_model: nn.Module,
                    sparsity_levels: List[int]) -> nn.Module:
        """Create model with specified sparsity levels"""
        model = base_model.clone()
        
        for i, (name, module) in enumerate(model.named_modules()):
            if i >= len(sparsity_levels):
                break
                
            level = self.sparsity_db.get_level(sparsity_levels[i])
            if level is None:
                continue
                
            # Apply sparsity configuration
            if "attention" in name.lower():
                self._prune_attention(module, level.attention_heads)
            elif "mlp" in name.lower():
                self._prune_mlp(module, level.mlp_dims)
                
        return model
        
    def _prune_attention(self, 
                        module: nn.Module,
                        num_heads: int):
        """Prune attention heads"""
        if not hasattr(module, "num_heads"):
            return
            
        # Keep only specified number of heads
        module.num_heads = num_heads
        head_dim = module.head_dim
        
        # Update projections
        if hasattr(module, "q_proj"):
            module.q_proj = nn.Linear(module.embed_dim, num_heads * head_dim)
        if not self.config.get("model", {}).get("is_gqa", False):
            if hasattr(module, "k_proj"):
                module.k_proj = nn.Linear(module.embed_dim, num_heads * head_dim)
            if hasattr(module, "v_proj"):
                module.v_proj = nn.Linear(module.embed_dim, num_heads * head_dim)
                
    def _prune_mlp(self,
                   module: nn.Module,
                   intermediate_size: int):
        """Prune MLP intermediate dimensions"""
        if not hasattr(module, "up_proj") or not hasattr(module, "down_proj"):
            return
            
        # Round to granularity
        granularity = self.config.get("mlp_granularity", 32)
        intermediate_size = (intermediate_size // granularity) * granularity
        
        # Update projections
        embed_dim = module.embed_dim
        module.up_proj = nn.Linear(embed_dim, intermediate_size)
        module.down_proj = nn.Linear(intermediate_size, embed_dim)
        if hasattr(module, "gate_proj"):
            module.gate_proj = nn.Linear(embed_dim, intermediate_size)
        
    def finetune(self,
                 model: nn.Module,
                 num_tokens: int,
                 batch_size: int = 32) -> nn.Module:
        """Finetune model on specified number of tokens"""
        model = model.to(self.device)
        model.train()
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Track tokens seen
        tokens_seen = 0
        step = 0
        
        while tokens_seen < num_tokens:
            for batch in self.train_loader:
                batch = batch.to(self.device)
                
                # Apply learning rate warmup
                if step < self.warmup_steps:
                    lr = self.learning_rate * (step / self.warmup_steps)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr
                
                # Forward pass
                outputs = model(batch)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                tokens_seen += batch.numel()
                step += 1
                
                if tokens_seen >= num_tokens:
                    break
                    
        model = model.cpu()
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
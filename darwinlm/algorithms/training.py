import torch
import torch.nn as nn
from typing import Dict, Any, List
from ..utils.metrics import compute_loss, compute_perplexity, ModelEvaluator
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

class TrainingAwareSelection:
    """Training-aware selection process with progressive evaluation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config["hardware"]["device"]
        
    def finetune_candidate(self, 
                          model: nn.Module,
                          tokens_schedule: List[int]) -> nn.Module:
        """Progressive finetuning of candidate model"""
        model = model.to(self.device)
        
        # Initialize optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=self.config["training"]["optimizer"]["lr"],
            weight_decay=self.config["training"]["optimizer"]["weight_decay"]
        )
        
        # Progressive finetuning
        for num_tokens in tokens_schedule:
            # Get training data for this step
            train_loader = self._get_training_data(num_tokens)
            
            # Create scheduler
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(len(train_loader) * 
                                   self.config["training"]["scheduler"]["warmup_ratio"]),
                num_training_steps=len(train_loader)
            )
            
            # Training loop
            model.train()
            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = compute_loss(outputs, batch["labels"])
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.config["training"]["max_grad_norm"]
                )
                
                optimizer.step()
                scheduler.step()
                
        return model
    
    def select_candidates(self,
                         models: List[nn.Module],
                         eval_tokens: int) -> List[int]:
        """Enhanced candidate selection with multiple criteria"""
        # Get evaluation data
        eval_loader = self._get_eval_data(eval_tokens)
        
        # Initialize evaluator
        evaluator = ModelEvaluator(self.config)
        
        # Evaluate each model
        metrics_list = []
        for model in models:
            model.eval()
            metrics = evaluator.compute_metrics(
                model,
                self.reference_model,
                eval_loader
            )
            metrics_list.append(metrics)
            
        # Compute weighted scores
        weights = self.config["evolution"]["selection_weights"]
        scores = []
        
        for metrics in metrics_list:
            score = (
                weights["perplexity"] * metrics["perplexity"] +
                weights["kl_divergence"] * metrics["kl_divergence"] +
                weights["hidden_similarity"] * (1 - metrics["hidden_similarity"]) +
                weights["attention_similarity"] * (1 - metrics["attention_similarity"]) +
                weights["compression"] * (1 / metrics["compression_ratio"])
            )
            scores.append(score)
            
        # Return indices sorted by score (lower is better)
        return sorted(range(len(scores)), key=lambda i: scores[i]) 
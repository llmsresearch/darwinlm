import torch
import random
import numpy as np
from typing import List, Tuple, Optional
from ..database.sparsity_db import SparsityDatabase
from ..training.trainer import Trainer
from ..utils.metrics import compute_kl_divergence

class EvolutionarySearch:
    """Training-aware evolutionary search for optimal pruning structure"""
    
    def __init__(self, 
                 sparsity_db: SparsityDatabase,
                 trainer: Trainer,
                 num_generations: int = 200,
                 offspring_size: int = 16,
                 selection_steps: int = 4,
                 finetune_tokens: Optional[List[int]] = None,
                 selection_tokens: Optional[List[int]] = None):
        self.db = sparsity_db
        self.trainer = trainer
        self.num_generations = num_generations
        self.offspring_size = offspring_size
        self.selection_steps = selection_steps
        
        # Default token schedules from paper if not provided
        self.finetune_tokens = finetune_tokens or [10_000, 50_000, 100_000, 200_000]
        self.selection_tokens = selection_tokens or [1024, 2048, 4096, 8192]
        
    def level_switch_mutation(self, parent: List[int]) -> List[int]:
        """Mutation operator that switches sparsity levels while preserving total"""
        offspring = parent.copy()
        
        # Randomly select modules (attention or MLP) to mutate
        module_type = random.choice(['attention', 'mlp'])
        
        # Get indices of modules of selected type
        if module_type == 'attention':
            indices = [i for i, level in enumerate(parent) if i % 2 == 0]
        else:
            indices = [i for i, level in enumerate(parent) if i % 2 == 1]
            
        if len(indices) < 2:
            return offspring
            
        # Randomly select two different positions
        pos1, pos2 = random.sample(indices, 2)
        
        # Increase level at pos1 and decrease at pos2 by same amount
        step = random.randint(1, min(self.db.num_levels - offspring[pos1], 
                                   offspring[pos2]))
        offspring[pos1] += step
        offspring[pos2] -= step
        
        return offspring
    
    def evaluate_fitness(self, 
                        model: torch.nn.Module, 
                        eval_tokens: int) -> float:
        """Compute KL divergence between pruned and dense model outputs"""
        return compute_kl_divergence(
            model,
            self.trainer.get_dense_model(),
            self.trainer.get_eval_batch(eval_tokens)
        )
        
    def search(self, 
               initial_model: torch.nn.Module,
               target_sparsity: float) -> Tuple[List[int], torch.nn.Module]:
        """Main evolutionary search loop with training-aware selection"""
        
        # Initialize with uniform sparsity levels
        num_layers = len(list(initial_model.modules()))
        parent = self.db.get_uniform_levels(target_sparsity, num_layers)
        
        best_fitness = float('inf')
        best_levels = None
        best_model = None
        
        for gen in range(self.num_generations):
            # Generate offspring through mutation
            candidates = [parent]
            for _ in range(self.offspring_size):
                offspring = self.level_switch_mutation(parent)
                candidates.append(offspring)
                
            # Multi-step selection process
            for step in range(self.selection_steps):
                candidate_models = []
                
                # Finetune each candidate
                for candidate in candidates:
                    model = self.trainer.stitch_model(initial_model, candidate)
                    model = self.trainer.finetune(
                        model, 
                        num_tokens=self.finetune_tokens[step],
                        batch_size=32  # As mentioned in paper
                    )
                    candidate_models.append(model)
                
                # Select best candidates based on KL divergence
                fitnesses = [
                    self.evaluate_fitness(model, self.selection_tokens[step])
                    for model in candidate_models
                ]
                
                # Sort by fitness (lower is better)
                sorted_idx = torch.argsort(torch.tensor(fitnesses))
                
                # Keep top half for next step
                num_survive = max(1, len(candidates) // 2)
                candidates = [candidates[i] for i in sorted_idx[:num_survive]]
                candidate_models = [candidate_models[i] for i in sorted_idx[:num_survive]]
                fitnesses = [fitnesses[i] for i in sorted_idx[:num_survive]]
                
            # Update parent with best candidate
            parent = candidates[0]
            
            # Track best solution
            if fitnesses[0] < best_fitness:
                best_fitness = fitnesses[0]
                best_levels = parent
                best_model = candidate_models[0]
                
        return best_levels, best_model 
import torch
import random
from typing import List, Tuple, Dict, Any
from .pruning import SecondOrderPruning, PruningMask
from .training import TrainingAwareSelection
from ..utils.metrics import compute_kl_divergence, compute_perplexity, compute_loss
from ..utils.logging import ExperimentLogger
from ..utils.checkpointing import CheckpointManager
from ..utils.visualization import EvolutionVisualizer

class EvolutionarySearch:
    """Training-aware evolutionary search for optimal pruning structure"""
    
    def __init__(self, 
                 model: torch.nn.Module,
                 pruner: SecondOrderPruning,
                 config: Dict[str, Any]):
        self.model = model
        self.pruner = pruner
        self.config = config
        self.logger = ExperimentLogger(config)
        self.checkpoint_manager = CheckpointManager(config)
        self.selector = TrainingAwareSelection(config)
        self.evaluator = None  # Assuming a ModelEvaluator instance is created elsewhere
        self.visualizer = EvolutionVisualizer(config["output_dir"])
        
        # Evolution parameters
        self.num_generations = config["evolution"]["num_generations"]
        self.population_size = config["evolution"]["offspring_size"]
        self.mutation_rate = config["evolution"]["mutation"]["rate"]
        self.crossover_rate = config["evolution"]["crossover"]["rate"]
        
        # Initialize population with random sparsity levels
        self.population = self._initialize_population()
        
    def _initialize_population(self) -> List[List[float]]:
        """Enhanced population initialization with layer-wise heuristics"""
        population = []
        num_layers = len(self.model.layers)
        
        # Strategy 1: Uniform sparsity
        sparsity_target = self.config["pruning"]["sparsity_target"]
        uniform_sparsity = [sparsity_target] * num_layers
        population.append(uniform_sparsity)
        
        # Strategy 2: Gradually increasing sparsity
        increasing_sparsity = [
            sparsity_target * (i + 1) / num_layers 
            for i in range(num_layers)
        ]
        increasing_sparsity = self._normalize_sparsity(increasing_sparsity)
        population.append(increasing_sparsity)
        
        # Strategy 3: Gradually decreasing sparsity
        decreasing_sparsity = [
            sparsity_target * (num_layers - i) / num_layers 
            for i in range(num_layers)
        ]
        decreasing_sparsity = self._normalize_sparsity(decreasing_sparsity)
        population.append(decreasing_sparsity)
        
        # Strategy 4: Alternating high/low sparsity
        alternating_sparsity = [
            sparsity_target * (1.5 if i % 2 == 0 else 0.5)
            for i in range(num_layers)
        ]
        alternating_sparsity = self._normalize_sparsity(alternating_sparsity)
        population.append(alternating_sparsity)
        
        # Fill remaining population with random strategies
        while len(population) < self.population_size:
            sparsity_levels = torch.rand(num_layers)
            sparsity_levels = self._normalize_sparsity(sparsity_levels.tolist())
            population.append(sparsity_levels)
            
        return population
    
    def _crossover(self, parent1: List[float], parent2: List[float]) -> List[float]:
        """Enhanced crossover with multiple strategies"""
        if random.random() > self.crossover_rate:
            return parent1
            
        strategy = random.choice([
            self._single_point_crossover,
            self._two_point_crossover,
            self._uniform_crossover,
            self._arithmetic_crossover
        ])
        
        return strategy(parent1, parent2)
    
    def _single_point_crossover(self, parent1: List[float], parent2: List[float]) -> List[float]:
        """Single-point crossover"""
        point = random.randint(0, len(parent1)-1)
        child = parent1[:point] + parent2[point:]
        return self._normalize_sparsity(child)
    
    def _two_point_crossover(self, parent1: List[float], parent2: List[float]) -> List[float]:
        """Two-point crossover"""
        points = sorted(random.sample(range(len(parent1)), 2))
        child = (parent1[:points[0]] + 
                parent2[points[0]:points[1]] + 
                parent1[points[1]:])
        return self._normalize_sparsity(child)
    
    def _uniform_crossover(self, parent1: List[float], parent2: List[float]) -> List[float]:
        """Uniform crossover"""
        child = [
            p1 if random.random() < 0.5 else p2
            for p1, p2 in zip(parent1, parent2)
        ]
        return self._normalize_sparsity(child)
    
    def _arithmetic_crossover(self, parent1: List[float], parent2: List[float]) -> List[float]:
        """Arithmetic crossover"""
        alpha = random.random()
        child = [
            alpha * p1 + (1 - alpha) * p2
            for p1, p2 in zip(parent1, parent2)
        ]
        return self._normalize_sparsity(child)
    
    def _mutate(self, individual: List[float]) -> List[float]:
        """Enhanced mutation with multiple strategies"""
        if random.random() > self.mutation_rate:
            return individual
            
        strategy = random.choice([
            self._gaussian_mutation,
            self._swap_mutation,
            self._inversion_mutation,
            self._scramble_mutation
        ])
        
        return strategy(individual)
    
    def _gaussian_mutation(self, individual: List[float]) -> List[float]:
        """Gaussian mutation"""
        mutated = torch.tensor(individual)
        noise = torch.randn_like(mutated) * self.config["evolution"]["mutation"]["sigma"]
        mutated += noise
        mutated.clamp_(0, 1)
        return self._normalize_sparsity(mutated.tolist())
    
    def _swap_mutation(self, individual: List[float]) -> List[float]:
        """Swap mutation"""
        mutated = individual.copy()
        idx1, idx2 = random.sample(range(len(individual)), 2)
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        return self._normalize_sparsity(mutated)
    
    def _inversion_mutation(self, individual: List[float]) -> List[float]:
        """Inversion mutation"""
        points = sorted(random.sample(range(len(individual)), 2))
        mutated = (individual[:points[0]] + 
                   list(reversed(individual[points[0]:points[1]])) + 
                   individual[points[1]:])
        return self._normalize_sparsity(mutated)
    
    def _scramble_mutation(self, individual: List[float]) -> List[float]:
        """Scramble mutation"""
        points = sorted(random.sample(range(len(individual)), 2))
        segment = individual[points[0]:points[1]]
        random.shuffle(segment)
        mutated = individual[:points[0]] + segment + individual[points[1]:]
        return self._normalize_sparsity(mutated)
    
    def _normalize_sparsity(self, sparsity_levels: List[float]) -> List[float]:
        """Normalize sparsity levels to maintain target"""
        sparsity_target = self.config["pruning"]["sparsity_target"]
        total = sum(sparsity_levels)
        return [s * sparsity_target * len(sparsity_levels) / total for s in sparsity_levels]
    
    def _evaluate_candidate(self, 
                          sparsity_levels: List[float],
                          eval_tokens: int) -> Tuple[torch.nn.Module, float]:
        """Evaluate candidate with comprehensive metrics"""
        # Get pruned model
        pruned_model = self._apply_sparsity(self.model, sparsity_levels)
        
        # Compute all metrics
        metrics = self.evaluator.compute_metrics(
            pruned_model,
            self.reference_model,
            self.eval_loader
        )
        
        # Compute weighted fitness score
        weights = self.config["evolution"]["fitness_weights"]
        fitness = (
            weights["perplexity"] * metrics["perplexity"] +
            weights["kl_divergence"] * metrics["kl_divergence"] +
            weights["robustness"] * (1 - metrics["robustness_score"]) +
            weights["memory"] * metrics["total_memory_mb"] / 1024  # Convert to GB
        )
        
        # Log metrics
        self.visualizer.log_generation(
            self.current_generation,
            metrics,
            sparsity_levels
        )
        
        return pruned_model, fitness
    
    def run(self) -> torch.nn.Module:
        """Main evolutionary search loop"""
        best_fitness = float('inf')
        best_model = None
        
        for generation in range(self.num_generations):
            # Generate offspring through crossover and mutation
            offspring = []
            while len(offspring) < self.population_size:
                parent1 = random.choice(self.population)
                parent2 = random.choice(self.population)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                offspring.append(child)
                
            # Evaluate all candidates
            candidates = []
            fitnesses = []
            for sparsity_levels in offspring:
                model, fitness = self._evaluate_candidate(
                    sparsity_levels,
                    self.config["evolution"]["finetune"]["selection_tokens"][-1]
                )
                candidates.append(model)
                fitnesses.append(fitness)
                
            # Selection
            combined = list(zip(offspring, candidates, fitnesses))
            combined.sort(key=lambda x: x[2])  # Sort by fitness
            
            # Update population
            self.population = [x[0] for x in combined[:self.population_size]]
            
            # Update best solution
            if combined[0][2] < best_fitness:
                best_fitness = combined[0][2]
                best_model = combined[0][1]
                
            # Logging and checkpointing
            self.logger.log_generation(generation, best_fitness)
            self.checkpoint_manager.save_checkpoint(
                best_model, generation, best_fitness, combined[0][0]
            )
            
        return best_model 
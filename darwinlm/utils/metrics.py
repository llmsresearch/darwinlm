import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
from transformers import PreTrainedModel
from torch.utils.data import DataLoader
import numpy as np

def compute_kl_divergence(model1: torch.nn.Module,
                         model2: torch.nn.Module,
                         input_ids: torch.Tensor,
                         attention_mask: Optional[torch.Tensor] = None,
                         temperature: float = 1.0) -> torch.Tensor:
    """Compute KL divergence between outputs of two models
    
    Args:
        model1: First model
        model2: Second model
        input_ids: Input token IDs
        attention_mask: Attention mask (optional)
        temperature: Temperature for softmax
        
    Returns:
        torch.Tensor: KL divergence value
    """
    # Ensure models are in eval mode
    model1.eval()
    model2.eval()
    
    with torch.no_grad():
        # Forward pass through both models
        outputs1 = model1(input_ids, attention_mask=attention_mask).logits
        outputs2 = model2(input_ids, attention_mask=attention_mask).logits
        
        # Apply temperature scaling
        if temperature != 1.0:
            outputs1 = outputs1 / temperature
            outputs2 = outputs2 / temperature
            
        # Compute probabilities
        probs1 = F.softmax(outputs1, dim=-1)
        probs2 = F.softmax(outputs2, dim=-1)
        
        # Compute KL divergence
        kl_div = F.kl_div(
            probs2.log(),
            probs1,
            reduction='batchmean',
            log_target=False
        )
        
    return kl_div

def compute_perplexity(model: torch.nn.Module,
                      input_ids: torch.Tensor,
                      attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute perplexity for a model
    
    Args:
        model: Model to evaluate
        input_ids: Input token IDs
        attention_mask: Attention mask (optional)
        
    Returns:
        torch.Tensor: Perplexity value
    """
    model.eval()
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = outputs.loss
        
    return torch.exp(loss)

def compute_model_size(model: torch.nn.Module) -> int:
    """Compute number of parameters in model
    
    Args:
        model: Model to analyze
        
    Returns:
        int: Number of parameters
    """
    return sum(p.numel() for p in model.parameters())

def compute_sparsity(model: torch.nn.Module) -> float:
    """Compute overall sparsity of model
    
    Args:
        model: Model to analyze
        
    Returns:
        float: Sparsity ratio (0-1)
    """
    total_params = 0
    zero_params = 0
    
    for p in model.parameters():
        total_params += p.numel()
        zero_params += (p == 0).sum().item()
        
    return zero_params / total_params

def compute_loss(outputs: Dict[str, torch.Tensor], 
                labels: torch.Tensor) -> torch.Tensor:
    """Compute model loss"""
    return F.cross_entropy(
        outputs.logits.view(-1, outputs.logits.size(-1)),
        labels.view(-1)
    )

def compute_hessian(inputs: torch.Tensor, 
                   outputs: torch.Tensor) -> torch.Tensor:
    """Compute Hessian matrix for OBS"""
    jacobian = torch.autograd.grad(outputs, inputs, 
                                 create_graph=True)[0]
    return torch.matmul(jacobian, jacobian.transpose(-1, -2))

class ModelEvaluator:
    """Comprehensive model evaluation metrics for LLM compression.
    
    This class provides a wide range of metrics to evaluate compressed models:
    1. Language Modeling Metrics:
       - Perplexity
       - Loss
       - Next Token Accuracy
       - Sequence Completion
       
    2. Behavioral Metrics:
       - KL Divergence from original model
       - Hidden State Similarity
       - Attention Pattern Similarity
       
    3. Efficiency Metrics:
       - Sparsity Ratio
       - Active Attention Heads
       - Active MLP Dimensions
       - Compression Ratio
       
    4. Robustness Metrics:
       - Perturbation Sensitivity
       - Input Noise Resilience
       
    5. Memory Metrics:
       - Parameter Memory Usage
       - Buffer Memory Usage
       - Total Memory Footprint
       
    6. Calibration Metrics:
       - Expected Calibration Error
       - Confidence Statistics
       
    Usage:
        evaluator = ModelEvaluator(config)
        metrics = evaluator.compute_metrics(
            model=compressed_model,
            reference_model=original_model,
            eval_loader=evaluation_dataloader
        )
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config["hardware"]["device"]
        
    def compute_metrics(self, 
                       model: PreTrainedModel,
                       reference_model: PreTrainedModel,
                       eval_loader: DataLoader) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics"""
        metrics = {}
        
        # Language modeling metrics
        metrics.update(self._compute_lm_metrics(model, eval_loader))
        
        # Behavioral metrics
        metrics.update(self._compute_behavioral_metrics(model, reference_model, eval_loader))
        
        # Efficiency metrics
        metrics.update(self._compute_efficiency_metrics(model))
        
        # Robustness metrics
        metrics.update(self._compute_robustness_metrics(model, eval_loader))
        
        # Memory metrics
        metrics.update(self._compute_memory_metrics(model))
        
        # Sequence metrics
        metrics.update(self._compute_sequence_metrics(model, eval_loader))
        
        # Calibration metrics
        metrics.update(self._compute_calibration_metrics(model, eval_loader))
        
        return metrics
        
    def _compute_lm_metrics(self, 
                           model: PreTrainedModel,
                           eval_loader: DataLoader) -> Dict[str, float]:
        """Compute language modeling metrics"""
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                
                # Perplexity and loss
                loss = F.cross_entropy(
                    outputs.logits.view(-1, outputs.logits.size(-1)),
                    batch["labels"].view(-1),
                    reduction='sum'
                )
                total_loss += loss.item()
                total_tokens += batch["labels"].ne(-100).sum().item()
                
        return {
            "perplexity": torch.exp(torch.tensor(total_loss / total_tokens)).item(),
            "loss": total_loss / total_tokens
        }
        
    def _compute_behavioral_metrics(self,
                                  model: PreTrainedModel,
                                  reference_model: PreTrainedModel,
                                  eval_loader: DataLoader) -> Dict[str, float]:
        """Compute behavioral similarity metrics"""
        kl_divs = []
        cosine_sims = []
        attention_sims = []
        
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Get outputs from both models
                model_outputs = model(**batch, output_attentions=True)
                ref_outputs = reference_model(**batch, output_attentions=True)
                
                # KL divergence on logits
                kl_div = F.kl_div(
                    F.log_softmax(model_outputs.logits, dim=-1),
                    F.softmax(ref_outputs.logits, dim=-1),
                    reduction='batchmean'
                )
                kl_divs.append(kl_div.item())
                
                # Hidden state similarity
                cosine_sim = F.cosine_similarity(
                    model_outputs.hidden_states[-1],
                    ref_outputs.hidden_states[-1],
                    dim=-1
                ).mean()
                cosine_sims.append(cosine_sim.item())
                
                # Attention pattern similarity
                for model_attn, ref_attn in zip(model_outputs.attentions,
                                              ref_outputs.attentions):
                    attention_sim = F.cosine_similarity(
                        model_attn.flatten(2),
                        ref_attn.flatten(2),
                        dim=-1
                    ).mean()
                    attention_sims.append(attention_sim.item())
                    
        return {
            "kl_divergence": sum(kl_divs) / len(kl_divs),
            "hidden_similarity": sum(cosine_sims) / len(cosine_sims),
            "attention_similarity": sum(attention_sims) / len(attention_sims)
        }
        
    def _compute_efficiency_metrics(self, model: PreTrainedModel) -> Dict[str, float]:
        """Compute model efficiency metrics"""
        total_params = sum(p.numel() for p in model.parameters())
        nonzero_params = sum(p.numel() for p in model.parameters() 
                            if (p != 0).sum() > 0)
        
        # Count attention heads and MLP dimensions
        num_heads = 0
        mlp_dims = 0
        for name, param in model.named_parameters():
            if "attention" in name and len(param.shape) == 3:
                num_heads += param.shape[1]
            elif "mlp" in name and len(param.shape) == 2:
                mlp_dims += param.shape[0]
                
        return {
            "sparsity": 1 - (nonzero_params / total_params),
            "active_heads": num_heads,
            "active_mlp_dims": mlp_dims,
            "compression_ratio": total_params / nonzero_params
        }
        
    def _compute_robustness_metrics(self,
                                  model: PreTrainedModel,
                                  eval_loader: DataLoader) -> Dict[str, float]:
        """Compute robustness metrics"""
        # Test model behavior under input perturbations
        original_perplexities = []
        perturbed_perplexities = []
        
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Get original outputs
                orig_outputs = model(**batch)
                orig_perp = torch.exp(compute_loss(orig_outputs, batch["labels"]))
                original_perplexities.append(orig_perp.item())
                
                # Add noise to inputs
                perturbed_inputs = batch["input_ids"].clone()
                mask = torch.rand_like(perturbed_inputs.float()) < 0.1
                perturbed_inputs[mask] = torch.randint_like(
                    perturbed_inputs[mask], 
                    0, 
                    self.config["model"]["vocab_size"]
                )
                
                # Get perturbed outputs
                batch["input_ids"] = perturbed_inputs
                pert_outputs = model(**batch)
                pert_perp = torch.exp(compute_loss(pert_outputs, batch["labels"]))
                perturbed_perplexities.append(pert_perp.item())
                
        return {
            "robustness_score": np.mean([
                o/p for o, p in zip(original_perplexities, perturbed_perplexities)
            ]),
            "perturbation_sensitivity": np.std([
                o/p for o, p in zip(original_perplexities, perturbed_perplexities)
            ])
        }
        
    def _compute_memory_metrics(self, model: PreTrainedModel) -> Dict[str, float]:
        """Compute memory usage metrics"""
        param_mem = sum(p.nelement() * p.element_size() 
                       for p in model.parameters()) / 1024**2  # MB
        buffer_mem = sum(b.nelement() * b.element_size() 
                        for b in model.buffers()) / 1024**2  # MB
        
        return {
            "parameter_memory_mb": param_mem,
            "buffer_memory_mb": buffer_mem,
            "total_memory_mb": param_mem + buffer_mem
        }
        
    def _compute_sequence_metrics(self,
                            model: PreTrainedModel,
                            eval_loader: DataLoader) -> Dict[str, float]:
        """Compute sequence-level metrics"""
        model.eval()
        next_token_accuracy = []
        sequence_completion = []
        
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Next token prediction
                outputs = model(**batch)
                preds = outputs.logits.argmax(dim=-1)
                accuracy = (preds == batch["labels"]).float().mean()
                next_token_accuracy.append(accuracy.item())
                
                # Sequence completion
                input_ids = batch["input_ids"][:, :input_ids.size(1)//2]
                target_ids = batch["input_ids"][:, input_ids.size(1)//2:]
                
                generated = model.generate(
                    input_ids,
                    max_length=target_ids.size(1),
                    num_return_sequences=1
                )
                
                completion_accuracy = (generated == target_ids).float().mean()
                sequence_completion.append(completion_accuracy.item())
                
        return {
            "next_token_accuracy": np.mean(next_token_accuracy),
            "sequence_completion": np.mean(sequence_completion)
        }
        
    def _compute_calibration_metrics(self,
                               model: PreTrainedModel,
                               eval_loader: DataLoader) -> Dict[str, float]:
        """Compute model calibration metrics"""
        confidences = []
        accuracies = []
        bins = np.linspace(0, 1, 11)
        
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                
                probs = F.softmax(outputs.logits, dim=-1)
                confidence, preds = probs.max(dim=-1)
                accuracy = (preds == batch["labels"]).float()
                
                confidences.extend(confidence.cpu().tolist())
                accuracies.extend(accuracy.cpu().tolist())
                
        # Compute ECE (Expected Calibration Error)
        confidences = np.array(confidences)
        accuracies = np.array(accuracies)
        
        ece = 0.0
        for i in range(len(bins)-1):
            bin_mask = (confidences >= bins[i]) & (confidences < bins[i+1])
            if bin_mask.any():
                bin_acc = accuracies[bin_mask].mean()
                bin_conf = confidences[bin_mask].mean()
                bin_size = bin_mask.mean()
                ece += bin_size * abs(bin_acc - bin_conf)
                
        return {
            "expected_calibration_error": ece,
            "mean_confidence": np.mean(confidences),
            "confidence_accuracy_correlation": np.corrcoef(confidences, accuracies)[0,1]
        } 
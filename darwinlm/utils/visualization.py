import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import pandas as pd
import torch
import numpy as np

class EvolutionVisualizer:
    """Advanced visualization tools for evolution progress and model analysis"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.metrics_history = []
        self.attention_history = []
        
    def log_generation(self, 
                      generation: int,
                      metrics: Dict[str, float],
                      sparsity_distribution: List[float]):
        """Log metrics for current generation"""
        metrics["generation"] = generation
        metrics["sparsity_distribution"] = sparsity_distribution
        self.metrics_history.append(metrics)
        
    def plot_metrics_dashboard(self):
        """Create comprehensive metrics dashboard"""
        df = pd.DataFrame(self.metrics_history)
        
        # Create 3x3 dashboard
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Performance Metrics
        ax1 = fig.add_subplot(gs[0, 0])
        sns.lineplot(data=df, x="generation", y="perplexity", ax=ax1)
        ax1.set_title("Perplexity Trend")
        
        ax2 = fig.add_subplot(gs[0, 1])
        sns.lineplot(data=df, x="generation", y="kl_divergence", ax=ax2)
        ax2.set_title("KL Divergence")
        
        ax3 = fig.add_subplot(gs[0, 2])
        sns.lineplot(data=df, x="generation", y="robustness_score", ax=ax3)
        ax3.set_title("Robustness Score")
        
        # Efficiency Metrics
        ax4 = fig.add_subplot(gs[1, 0])
        sns.lineplot(data=df, x="generation", y="sparsity", ax=ax4)
        ax4.set_title("Model Sparsity")
        
        ax5 = fig.add_subplot(gs[1, 1])
        sns.lineplot(data=df, x="generation", y="compression_ratio", ax=ax5)
        ax5.set_title("Compression Ratio")
        
        ax6 = fig.add_subplot(gs[1, 2])
        sns.lineplot(data=df, x="generation", y="total_memory_mb", ax=ax6)
        ax6.set_title("Memory Usage (MB)")
        
        # Behavioral Metrics
        ax7 = fig.add_subplot(gs[2, 0])
        sns.lineplot(data=df, x="generation", y="hidden_similarity", ax=ax7)
        ax7.set_title("Hidden State Similarity")
        
        ax8 = fig.add_subplot(gs[2, 1])
        sns.lineplot(data=df, x="generation", y="attention_similarity", ax=ax8)
        ax8.set_title("Attention Pattern Similarity")
        
        ax9 = fig.add_subplot(gs[2, 2])
        sns.lineplot(data=df, x="generation", y="perturbation_sensitivity", ax=ax9)
        ax9.set_title("Perturbation Sensitivity")
        
        plt.savefig(f"{self.output_dir}/metrics_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_attention_evolution(self, layer_idx: int, head_idx: int):
        """Visualize attention pattern evolution over generations"""
        if not self.attention_history:
            return
            
        num_gens = len(self.attention_history)
        fig, axes = plt.subplots(1, num_gens, figsize=(5*num_gens, 5))
        
        for gen, attn_maps in enumerate(self.attention_history):
            attn_map = attn_maps[layer_idx, head_idx].cpu().numpy()
            sns.heatmap(attn_map, ax=axes[gen], cmap='viridis')
            axes[gen].set_title(f"Generation {gen}")
            
        plt.suptitle(f"Attention Evolution (Layer {layer_idx}, Head {head_idx})")
        plt.savefig(f"{self.output_dir}/attention_evolution_l{layer_idx}_h{head_idx}.png")
        plt.close()
        
    def plot_layer_importance(self, importance_scores: torch.Tensor):
        """Visualize layer importance scores"""
        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(range(len(importance_scores))), 
                   y=importance_scores.cpu().numpy())
        plt.title("Layer Importance Distribution")
        plt.xlabel("Layer Index")
        plt.ylabel("Importance Score")
        plt.savefig(f"{self.output_dir}/layer_importance.png")
        plt.close()
        
    def plot_sparsity_heatmap(self):
        """Create heatmap of sparsity patterns across layers and generations"""
        sparsity_matrix = np.array([
            m["sparsity_distribution"] for m in self.metrics_history
        ])
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(sparsity_matrix, cmap='YlOrRd')
        plt.title("Sparsity Evolution Across Layers")
        plt.xlabel("Layer Index")
        plt.ylabel("Generation")
        plt.savefig(f"{self.output_dir}/sparsity_heatmap.png")
        plt.close()
        
    def generate_html_report(self):
        """Generate HTML report with all visualizations and metrics"""
        template = """
        <html>
        <head>
            <title>DarwinLM Evolution Report</title>
            <style>
                body { font-family: Arial; margin: 20px; }
                .metric-card { 
                    border: 1px solid #ddd; 
                    padding: 15px;
                    margin: 10px;
                    border-radius: 5px;
                }
                .plot-container { margin: 20px 0; }
            </style>
        </head>
        <body>
            <h1>DarwinLM Evolution Report</h1>
            
            <h2>Final Metrics</h2>
            <div class="metrics-container">
                {metrics_html}
            </div>
            
            <h2>Evolution Plots</h2>
            <div class="plot-container">
                <img src="metrics_dashboard.png" width="100%">
                <img src="sparsity_heatmap.png" width="100%">
            </div>
            
            <h2>Layer Analysis</h2>
            <div class="plot-container">
                <img src="layer_importance.png" width="100%">
            </div>
        </body>
        </html>
        """
        
        # Generate metrics HTML
        final_metrics = self.metrics_history[-1]
        metrics_html = ""
        for metric, value in final_metrics.items():
            if metric != "sparsity_distribution":
                metrics_html += f"""
                <div class="metric-card">
                    <h3>{metric}</h3>
                    <p>{value:.4f}</p>
                </div>
                """
                
        # Save HTML report
        with open(f"{self.output_dir}/evolution_report.html", 'w') as f:
            f.write(template.format(metrics_html=metrics_html)) 
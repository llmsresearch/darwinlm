from .models.adapter import ModelAdapter
from .algorithms.pruning import SecondOrderPruning
from .algorithms.evolution import EvolutionarySearch
from .algorithms.training import TrainingAwareSelection
from .data.manager import DataManager
from .utils.config import load_config

__version__ = "0.1.0"

def compress_model(config_path: str, model: str = None, **overrides):
    """Main entry point for model compression"""
    # Load configuration
    if model:
        overrides["defaults"] = [{"model": model}]
    config = load_config(config_path, **overrides)
    
    # Initialize components
    model_adapter = ModelAdapter(config)
    data_manager = DataManager(config)
    
    # Load model
    model = model_adapter.load_model()
    
    # Initialize algorithms
    pruner = SecondOrderPruning(model, config)
    search = EvolutionarySearch(
        model=model,
        pruner=pruner,
        config=config
    )
    
    # Run compression
    compressed_model = search.run()
    return compressed_model 
import yaml
from typing import Dict, Any

from .models.model_adapter import ModelAdapter
from .algorithms.pruning import SecondOrderPruning
from .algorithms.evolution import EvolutionarySearch
from .algorithms.training import TrainingAwareSelection
from .data.data_manager import DataManager
from .utils.logging import ExperimentLogger
from .utils.visualization import EvolutionVisualizer

class DarwinLM:
    """Main class for DarwinLM model compression"""
    
    def __init__(self, config_path: str):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Initialize components
        self.model_adapter = ModelAdapter(self.config)
        self.data_manager = DataManager(self.config)
        self.logger = ExperimentLogger(self.config)
        self.visualizer = EvolutionVisualizer(self.config["output_dir"])
        
    def compress(self) -> None:
        """Main compression pipeline"""
        # Load model and data
        model = self.model_adapter.load_model()
        calibration_loader = self.data_manager.get_calibration_loader()
        
        # Initialize components
        pruner = SecondOrderPruning(
            model=model,
            config=self.config,
            calibration_data=next(iter(calibration_loader))
        )
        
        # Initialize evolutionary search
        search = EvolutionarySearch(
            model=model,
            pruner=pruner,
            config=self.config,
            logger=self.logger,
            visualizer=self.visualizer
        )
        
        # Run evolutionary search
        best_model = search.run()
        
        return best_model

if __name__ == "__main__":
    darwinlm = DarwinLM("config/default.yaml")
    compressed_model = darwinlm.compress() 
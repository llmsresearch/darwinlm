from typing import Dict, Any
import os
from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.core.config_store import ConfigStore

@dataclass
class ModelConfig:
    name: str
    pretrained_path: str
    architecture: str
    attention: Dict[str, Any]
    quantization: Dict[str, bool]

@dataclass
class TrainingConfig:
    optimizer: Dict[str, Any]
    scheduler: Dict[str, Any]
    max_grad_norm: float
    batch_size: int

@dataclass
class DarwinLMConfig:
    model: ModelConfig
    training: TrainingConfig
    hardware: Dict[str, Any]
    pruning: Dict[str, Any]
    evolution: Dict[str, Any]
    data: Dict[str, Any]
    seed: int
    output_dir: str
    log_level: str

cs = ConfigStore.instance()
cs.store(name="base_config", node=DarwinLMConfig)

def load_config(config_path: str, **overrides) -> DictConfig:
    """Load and validate configuration"""
    base_conf = OmegaConf.load(config_path)
    
    # Apply overrides
    override_conf = OmegaConf.create(overrides)
    conf = OmegaConf.merge(base_conf, override_conf)
    
    # Validate paths
    conf.output_dir = os.path.expanduser(conf.output_dir)
    conf.data.train.path = os.path.expanduser(conf.data.train.path)
    
    return conf

def save_config(config: DictConfig, path: str):
    """Save configuration to file"""
    OmegaConf.save(config, path) 
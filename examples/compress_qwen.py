import hydra
from omegaconf import DictConfig
from darwinlm import compress_model

@hydra.main(config_path="../configs", config_name="base")
def main(cfg: DictConfig):
    # Override model config
    cfg.defaults[0] = {"model": "qwen_14b"}
    
    # Run compression
    compressed_model = compress_model(cfg)
    
    # Save compressed model
    compressed_model.save_pretrained(cfg.output_dir)

if __name__ == "__main__":
    main() 
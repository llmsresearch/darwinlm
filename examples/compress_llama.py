import hydra
from omegaconf import DictConfig
from darwinlm import compress_model

@hydra.main(config_path="../configs", config_name="base")
def main(cfg: DictConfig):
    """Compress Llama2 model"""
    # Run compression
    compressed_model = compress_model(cfg, model="llama2_7b")
    
    # Save compressed model
    compressed_model.save_pretrained(cfg.output_dir)

if __name__ == "__main__":
    main() 
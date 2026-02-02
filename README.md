# DarwinLM: Evolutionary Structured Pruning of Large Language Models

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://llmsresearch.github.io/darwinlm)
[![arXiv](https://img.shields.io/badge/arXiv-2502.07780-b31b1b.svg)](https://arxiv.org/abs/2502.07780)

This repository contains the unofficial implementation of [DarwinLM: Evolutionary Structured Pruning of Large Language Models](https://arxiv.org/abs/2502.07780).

> This implementation is part of [LLMs Research](https://llmsresearch.substack.com), where I break down ~100 papers weekly and build the ones worth building.

---
## Project Structure
```
darwinlm/
├── configs/                  # Configuration files
│   ├── models/              # Model-specific configs
│   │   ├── llama2_7b.yaml
│   │   ├── qwen_14b.yaml
│   │   └── mistral_7b.yaml
│   └── base.yaml            # Base configuration
├── darwinlm/                # Main package
│   ├── algorithms/          # Core algorithms
│   │   ├── pruning.py      # Second-order pruning
│   │   ├── evolution.py    # Evolutionary search
│   │   └── training.py     # Training-aware selection
│   ├── models/             # Model handling
│   │   └── adapter.py      # Model adapter
│   ├── data/               # Data processing
│   │   └── manager.py      # Data manager
│   └── utils/              # Utilities
│       ├── config.py       # Config management
│       ├── logging.py      # Logging utilities
│       ├── checkpointing.py
│       └── metrics.py      # Evaluation metrics
├── tests/                  # Test suite
├── examples/               # Example scripts
├── setup.py               # Package setup
└── README.md              # Documentation
```

## Installation

```bash
# Clone repository
git clone https://github.com/llmsresearch/darwinlm.git
cd darwinlm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Usage

1. Configure your model:
```yaml
# configs/models/your_model.yaml
model:
  name: "your-model"
  pretrained_path: "path/to/model"
  architecture: "llama"  # or other supported architectures
```

2. Run compression:
```bash
# Compress Llama2 model
python examples/compress_llama.py

# Compress Qwen model
python examples/compress_qwen.py

# Compress with custom config
python examples/compress_llama.py model=llama2_7b pruning.sparsity_target=0.6
```

3. The compressed model will be saved in the specified output directory.

## Configuration

The configuration system uses Hydra and is organized as follows:

- `configs/base.yaml`: Base configuration with default values
- `configs/models/`: Model-specific configurations
  - `llama2_7b.yaml`: Llama 2 7B configuration
  - `qwen_14b.yaml`: Qwen 14B configuration
  - `mistral_7b.yaml`: Mistral 7B configuration

You can override any config value through command line:
```bash
python examples/compress_llama.py \
    model=llama2_7b \
    pruning.sparsity_target=0.6 \
    training.batch_size=64
```

## Contributing

We welcome contributions! Please check our [Contributing Guidelines](CONTRIBUTING.md) for details.

Some areas we'd love help with:
- Adding support for new model architectures
- Implementing alternative pruning strategies
- Improving evolution operators
- Enhancing documentation and examples

## Citation

```bibtex
@article{tang2025darwinlm,
  title={DarwinLM: Evolutionary Structured Pruning of Large Language Models},
  author={Tang, Shengkun and Sieberling, Oliver and Kurtic, Eldar and Shen, Zhiqiang and Alistarh, Dan},
  journal={arXiv preprint arXiv:2502.07780},
  year={2025}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

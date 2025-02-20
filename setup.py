from setuptools import setup, find_packages

setup(
    name="darwinlm",
    version="0.1.0",
    description="Evolutionary Structured Pruning of Large Language Models",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "wandb>=0.15.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="llm, compression, evolutionary-algorithms, model-pruning",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/darwinlm/issues",
        "Source": "https://github.com/yourusername/darwinlm",
    },
) 
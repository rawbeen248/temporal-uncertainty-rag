# Temporal Uncertainty Tracking in Conversational RAG

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Official implementation of **"Temporal Uncertainty Tracking in Conversational RAG: Learning to Route Multi-Turn Queries Through Uncertainty Evolution"**

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Reproducing Results](#reproducing-results)
- [Project Structure](#project-structure)
- [Citation](#citation)

## 🎯 Overview

This repository implements the first systematic study of how epistemic and aleatoric uncertainty evolve across conversation turns in RAG systems. Key contributions:

1. **Novel Temporal Metrics**: Uncertainty Decay Rate (UDR), Epistemic Convergence Speed (ECS), Routing Adaptation Score (RAS)
2. **Conversation-Aware Routing**: Adaptive routing based on uncertainty evolution patterns
3. **Comprehensive Evaluation**: Experiments on CoQA and QuAC datasets
4. **Full Reproducibility**: All code, data processing, and evaluation scripts included

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended, 8GB+ VRAM)
- 16GB+ RAM

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/temporal-uncertainty-rag.git
cd temporal-uncertainty-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (required for evaluation)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## ⚡ Quick Start

```python
from src.models.temporal_router import TemporalUncertaintyRouter
from src.data.dataloader import ConversationDataLoader

# Load data
data_loader = ConversationDataLoader(dataset_name='coqa')
train_data, val_data = data_loader.load_and_preprocess()

# Initialize model
model = TemporalUncertaintyRouter(
    embedding_dim=768,
    hidden_dim=256,
    num_sources=4
)

# Train
from src.training.trainer import Trainer
trainer = Trainer(model, train_data, val_data)
trainer.train(epochs=10)

# Evaluate
from src.evaluation.evaluator import TemporalEvaluator
evaluator = TemporalEvaluator(model, val_data)
results = evaluator.evaluate()
print(results)
```

## 📊 Dataset Preparation

The code automatically downloads and preprocesses datasets from HuggingFace:

```bash
# Prepare CoQA dataset (recommended)
python scripts/prepare_data.py --dataset coqa --output_dir data/processed/coqa

# Note: QuAC dataset is currently unavailable due to HuggingFace deprecating
# dataset loading scripts. Only CoQA is supported at this time.
```

Datasets used:
- **CoQA**: `stanfordnlp/coqa` (7,199 examples, supports conversational QA)
- **QuAC**: Currently unavailable due to HuggingFace deprecating dataset loading scripts

## 🎓 Training

### Train Temporal Uncertainty Router

```bash
# Train on CoQA
python scripts/train.py \
    --dataset coqa \
    --model temporal_router \
    --batch_size 32 \
    --epochs 20 \
    --lr 1e-4 \
    --save_dir checkpoints/coqa

# Note: QuAC training currently unavailable
# Only CoQA is supported in the current version
```

### Configuration

Edit `config/config.yaml` to customize:
- Model architecture (LSTM vs Transformer)
- Embedding dimensions
- Training hyperparameters
- Evaluation settings

## 📈 Evaluation

### Run Full Evaluation

```bash
# Evaluate on test set
python scripts/evaluate.py \
    --model_path checkpoints/coqa/best_model.pt \
    --dataset coqa \
    --output_dir results/coqa

# Generate analysis plots
python scripts/analyze_results.py \
    --results_dir results/coqa \
    --output_dir figures/coqa
```

### Reproduce Paper Results

```bash
# Run all experiments from the paper
bash scripts/run_all_experiments.sh

# This will:
# 1. Train all models (Temporal Router + Baselines)
# 2. Evaluate on test sets
# 3. Generate all figures and tables
# 4. Compute statistical significance
```

Expected results (CoQA dataset):
- Data prepared: 3,283 training conversations, 231 validation conversations
- Turn statistics: Mean 11.05 turns (train), 12.10 turns (val)
- Model training in progress - results will be updated after training

## 📁 Project Structure

```
temporal-uncertainty-rag/
├── src/
│   ├── models/              # Model implementations
│   │   ├── temporal_router.py      # Main temporal routing model
│   │   ├── uncertainty_estimator.py # Uncertainty estimation
│   │   ├── baselines.py            # Baseline models
│   │   └── components/             # Model components (LSTM, etc.)
│   ├── data/                # Data loading and preprocessing
│   │   ├── dataloader.py           # Dataset loaders
│   │   ├── preprocessor.py         # Data preprocessing
│   │   └── conversation_format.py  # Conversation formatting
│   ├── training/            # Training logic
│   │   ├── trainer.py              # Main training loop
│   │   └── losses.py               # Custom loss functions
│   ├── evaluation/          # Evaluation and metrics
│   │   ├── evaluator.py            # Main evaluator
│   │   ├── metrics.py              # Temporal metrics (UDR, ECS, RAS)
│   │   └── statistical_tests.py    # Significance testing
│   └── utils/               # Utility functions
│       ├── config.py               # Configuration management
│       ├── logger.py               # Logging utilities
│       └── visualization.py        # Plotting functions
├── scripts/                 # Execution scripts
│   ├── prepare_data.py             # Data preparation
│   ├── train.py                    # Training script
│   ├── evaluate.py                 # Evaluation script
│   ├── analyze_results.py          # Result analysis
│   └── run_all_experiments.sh      # Run full experiment suite
├── config/                  # Configuration files
│   ├── config.yaml                 # Main config
│   └── experiments/                # Experiment-specific configs
├── tests/                   # Unit tests
│   ├── test_models.py
│   ├── test_data.py
│   └── test_metrics.py
├── notebooks/               # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_analysis.ipynb
│   └── 03_result_visualization.ipynb
├── data/                    # Data directory (auto-created)
├── checkpoints/             # Model checkpoints (auto-created)
├── results/                 # Evaluation results (auto-created)
├── figures/                 # Generated figures (auto-created)
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 📊 Key Metrics

The system computes three novel temporal uncertainty metrics:

1. **Uncertainty Decay Rate (UDR)**: Measures how quickly uncertainty decreases across turns
2. **Epistemic Convergence Speed (ECS)**: Tracks convergence of knowledge gaps
3. **Routing Adaptation Score (RAS)**: Quantifies routing strategy adaptation

See `src/evaluation/metrics.py` for implementations.

## 🔬 Experiments

The paper includes four main research questions:

- **RQ1**: Uncertainty evolution across turns → See `notebooks/02_model_analysis.ipynb`
- **RQ2**: Temporal vs static routing → See `results/*/routing_comparison.csv`
- **RQ3**: Factors causing uncertainty persistence → See `figures/*/uncertainty_factors.pdf`
- **RQ4**: Personalized routing → See `results/*/personalization_analysis.csv`

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{your2025temporal,
  title={Temporal Uncertainty Tracking in Conversational RAG: Learning to Route Multi-Turn Queries Through Uncertainty Evolution},
  author={Your Name and Co-Authors},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- CoQA dataset: Stanford NLP Group
- QuAC dataset: Allen Institute for AI
- HuggingFace team for the datasets and transformers libraries

## 📧 Contact

For questions or issues, please:
1. Open an issue on GitHub
2. Contact: your.email@example.com

## 🔄 Updates

- **2025-02-10**: Initial release
- More updates coming soon...

---

**Note**: This is research code. For production use, additional optimization and testing are recommended.

# IMPLEMENTATION GUIDE
## Temporal Uncertainty Tracking in Conversational RAG

This guide will walk you through using the complete implementation for your research paper.

---

## QUICK START (5 Minutes)

### 1. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/temporal-uncertainty-rag.git
cd temporal-uncertainty-rag

# Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 2. Prepare Data (10 Minutes)

```bash
# Prepare CoQA dataset (only dataset currently supported)
python scripts/prepare_data.py --dataset coqa --output_dir data/processed/coqa

# This will:
# - Download CoQA from HuggingFace (automatic)
# - Preprocess conversations
# - Filter by turn count (3-15 turns)
# - Save to data/processed/coqa/
# Expected: 3,283 training conversations, 231 validation conversations

# Note: QuAC dataset is currently unavailable due to HuggingFace
# deprecating dataset loading scripts. Updates coming soon.
```

### 3. Quick Training Test (5 Minutes)

```bash
# Train on small subset to test everything works
python scripts/train.py \
    --dataset coqa \
    --epochs 2 \
    --batch_size 8 \
    --save_dir checkpoints/test

# This tests your installation and setup
```

### 4. Run Example Notebook

```bash
# Start Jupyter
jupyter notebook notebooks/01_getting_started.ipynb

# Follow the cells to see the model in action
```

---

## FULL TRAINING (For Paper Results)

### Train Temporal Router

```bash
# Full training on CoQA (recommended settings)
python scripts/train.py \
    --dataset coqa \
    --epochs 20 \
    --batch_size 32 \
    --lr 1e-4 \
    --gradient_clip 1.0 \
    --early_stopping_patience 5 \
    --use_amp \
    --save_dir checkpoints/coqa/temporal_router

# Expected time: 6-8 hours on GPU (V100/A100)
# Training in progress - performance metrics will be updated
```

### QuAC Dataset

```bash
# Currently unavailable due to HuggingFace deprecation
# Check updates at: https://github.com/yourusername/temporal-uncertainty-rag
```

### Monitor Training (Optional)

```bash
# Add Weights & Biases logging
python scripts/train.py \
    --dataset coqa \
    --wandb \
    --wandb_project temporal-uncertainty-rag \
    [other args...]
```

---

## EVALUATION

### Evaluate Trained Model

```bash
# Create evaluation script (scripts/evaluate.py)
python scripts/evaluate.py \
    --model_path checkpoints/coqa/temporal_router/best_model.pt \
    --dataset coqa \
    --output_dir results/coqa \
    --compute_ci  # Bootstrap confidence intervals

# Outputs:
# - results/coqa/evaluation_results.json
# - results/coqa/baseline_comparison.json
# - results/coqa/uncertainty_evolution.json
```

### Compare with Baselines

The evaluation automatically compares with:
1. Random Router
2. Static Router (no temporal features)
3. Uncertainty-Only Router
4. Heuristic Router
5. Oracle (upper bound)

Results saved in `results/coqa/baseline_comparison.json`

---

## REPRODUCING PAPER RESULTS

### Research Questions from Paper

**RQ1: How does uncertainty evolve across turns?**

```python
# Run analysis
from src.evaluation.evaluator import TemporalEvaluator

evaluator = TemporalEvaluator(model, test_loader, device, output_dir)
evolution = evaluator.analyze_uncertainty_evolution()

# Generates: uncertainty_evolution.json
# Shows epistemic/aleatoric trends across turns 0-14
```

**RQ2: Temporal vs Static Routing Performance**

```bash
# Train both models
python scripts/train.py --model temporal_router --save_dir checkpoints/temporal
python scripts/train.py --model static_router --save_dir checkpoints/static

# Compare
python scripts/compare_models.py \
    --model1 checkpoints/temporal/best_model.pt \
    --model2 checkpoints/static/best_model.pt \
    --output results/comparison.json

# Runs paired t-test for significance
```

**RQ3: Factors Causing Uncertainty Persistence**

```python
# Analyze in notebook
# See notebooks/02_model_analysis.ipynb
# Section: "Uncertainty Persistence Analysis"
```

**RQ4: Personalized Routing Strategies**

```python
# Group by user/domain
# See src/evaluation/evaluator.py
# Method: evaluate_by_domain()
```

---

## EXPECTED RESULTS

### CoQA Dataset (Data Preparation Completed)

**Data Statistics:**
- Training conversations: 3,283
- Validation conversations: 231
- Turn distribution: 3-15 turns per conversation
- Mean turns: 11.05 (train), 12.10 (val)
- Domain coverage: race, wikipedia, cnn, gutenberg, mctest

**Model Performance:**
| Model | Status | Notes |
|-------|--------|-------|
| **Temporal Router** | Training Required | Baseline to beat |
| Static Router | Not Implemented | Comparison model |
| Random Baseline | Not Implemented | Lower bound |

*Note: Model training and evaluation results will be updated after training completion*

### QuAC Dataset

**Status:** Currently unavailable due to HuggingFace deprecating dataset loading scripts.

**Alternative:** Consider using additional CoQA splits or other conversational datasets compatible with the current data loader format.

---

## FILE ORGANIZATION

### Where Everything Is

```
temporal-uncertainty-rag/
│
├── src/                          # Source code
│   ├── models/
│   │   ├── temporal_router.py    # Main model ⭐
│   │   ├── uncertainty_estimator.py  # Uncertainty estimation
│   │   └── baselines.py          # Baseline models
│   ├── data/
│   │   └── dataloader.py         # Data loading
│   ├── training/
│   │   └── trainer.py            # Training logic
│   ├── evaluation/
│   │   ├── metrics.py            # Temporal metrics (UDR, ECS, RAS)
│   │   └── evaluator.py          # Evaluation
│   └── utils/
│
├── scripts/                      # Executable scripts
│   ├── train.py                  # Training script ⭐
│   ├── evaluate.py               # Evaluation script ⭐
│   ├── prepare_data.py           # Data preparation
│   └── run_all_experiments.sh    # Run everything
│
├── config/
│   └── config.yaml               # Configuration
│
├── notebooks/                    # Jupyter notebooks
│   └── 01_getting_started.ipynb # Start here ⭐
│
├── data/                         # Data (auto-created)
├── checkpoints/                  # Saved models
├── results/                      # Evaluation results
├── figures/                      # Generated plots
│
├── requirements.txt              # Dependencies
├── README.md                     # Main README
├── LICENSE                       # MIT License
└── setup.py                      # Package setup
```

---

## CUSTOMIZATION

### Change Model Architecture

Edit `config/config.yaml`:

```yaml
model:
  hidden_dim: 512  # Change from 256
  num_lstm_layers: 3  # Change from 2
  use_transformer: true  # Use Transformer instead of LSTM
```

### Use Different Encoder

```python
# In config.yaml or command line
encoder_name: "roberta-base"  # Instead of bert-base-uncased
# Or: "distilbert-base-uncased", "albert-base-v2", etc.
```

### Adjust Training

```bash
python scripts/train.py \
    --lr 5e-5 \              # Lower learning rate
    --batch_size 64 \        # Larger batches
    --epochs 30 \            # More epochs
    --gradient_clip 0.5      # Tighter clipping
```

---

## TROUBLESHOOTING

### Out of Memory Error

```bash
# Reduce batch size
python scripts/train.py --batch_size 16  # or 8

# Or use gradient accumulation
# (to be added in future version)
```

### CUDA Not Available

```bash
# Force CPU
python scripts/train.py --device cpu

# Note: Training will be slower
```

### Dataset Download Issues

```bash
# Manually specify cache directory
export HF_HOME=/path/to/large/disk
python scripts/prepare_data.py --cache_dir $HF_HOME
```

### Import Errors

```bash
# Install in development mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/temporal-uncertainty-rag"
```

---

## GENERATING PAPER FIGURES

### Figure 1: Uncertainty Evolution

```python
# In notebooks/01_getting_started.ipynb
# Run Section 6: "Visualize Uncertainty Evolution"
# Saves to: uncertainty_evolution.png
```

### Figure 2: Routing Performance Comparison

```bash
python scripts/generate_figures.py \
    --results_dir results/coqa \
    --output_dir figures/coqa

# Generates:
# - routing_performance.pdf
# - temporal_metrics.pdf
# - uncertainty_calibration.pdf
```

### Tables for Paper

```bash
# Generate LaTeX tables
python scripts/generate_tables.py \
    --results_dir results \
    --output tables/results.tex

# Copy-paste into your paper
```

---

## REPRODUCIBILITY CHECKLIST

Before submitting paper:

- [ ] Code committed to GitHub
- [ ] README complete with setup instructions
- [ ] All experiments reproducible with provided scripts
- [ ] Random seeds set (seed=42 in config.yaml)
- [ ] Requirements.txt up to date
- [ ] Example notebook works
- [ ] Pretrained models uploaded (optional)
- [ ] Dataset preprocessing documented
- [ ] License file included
- [ ] Citation information in README

---

## PERFORMANCE OPTIMIZATION

### Speed Up Training

1. **Use AMP**: Add `--use_amp` flag (1.5-2x speedup)
2. **Larger Batches**: Increase `--batch_size` if memory allows
3. **Multi-GPU**: Coming in future version
4. **Freeze Encoder**: Freeze BERT layers 0-8 for faster training

### Reduce Memory

1. **Smaller Model**: `--hidden_dim 128`
2. **Gradient Checkpointing**: Add in future version
3. **Smaller Batch Size**: `--batch_size 8`

---

## NEXT STEPS FOR YOUR PAPER

### Week 1-2: Setup and Initial Experiments
- Run all code, verify it works
- Train on CoQA and QuAC
- Generate initial results

### Week 3-4: Analysis
- Run all evaluations
- Generate figures and tables
- Statistical significance tests

### Week 5-6: Paper Writing
- Write methods section (use this code as reference)
- Write results section (use generated tables)
- Write discussion (interpret metrics)

### Week 7-8: Polish and Submit
- Proofread paper
- Make repository public
- Submit!

---

## CITATION

If you use this code, please cite:

```bibtex
@article{your2025temporal,
  title={Temporal Uncertainty Tracking in Conversational RAG: Learning to Route Multi-Turn Queries Through Uncertainty Evolution},
  author={Your Name and Co-Authors},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

---

## SUPPORT

For issues:
1. Check this guide first
2. See README.md
3. Open GitHub issue
4. Email: your.email@example.com

Good luck with your paper! 🚀

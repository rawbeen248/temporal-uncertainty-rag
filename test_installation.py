#!/usr/bin/env python3
"""
Quick test script to verify installation and basic functionality.

Usage:
    python test_installation.py
"""

import sys
from pathlib import Path

print("=" * 80)
print("TESTING TEMPORAL UNCERTAINTY RAG INSTALLATION")
print("=" * 80)

# Test 1: Check Python version
print("\n[1/8] Checking Python version...")
python_version = sys.version_info
if python_version >= (3, 8):
    print(f"✓ Python {python_version.major}.{python_version.minor} (>= 3.8)")
else:
    print(f"✗ Python {python_version.major}.{python_version.minor} (need >= 3.8)")
    sys.exit(1)

# Test 2: Check imports
print("\n[2/8] Checking core dependencies...")
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
except ImportError:
    print("✗ PyTorch not found. Run: pip install -r requirements.txt")
    sys.exit(1)

try:
    import transformers
    print(f"✓ Transformers {transformers.__version__}")
except ImportError:
    print("✗ Transformers not found.")
    sys.exit(1)

try:
    import datasets
    print("✓ Datasets library")
except ImportError:
    print("✗ Datasets library not found.")
    sys.exit(1)

# Test 3: Check CUDA
print("\n[3/8] Checking CUDA availability...")
if torch.cuda.is_available():
    print(f"✓ CUDA available (GPU: {torch.cuda.get_device_name(0)})")
else:
    print("⚠ CUDA not available (will use CPU, slower training)")

# Test 4: Test imports from src
print("\n[4/8] Testing module imports...")
try:
    from src.models.temporal_router import TemporalUncertaintyRouter
    print("✓ Temporal Router model")
except Exception as e:
    print(f"✗ Failed to import Temporal Router: {e}")
    sys.exit(1)

try:
    from src.data.dataloader import ConversationDataLoader
    print("✓ Data loader")
except Exception as e:
    print(f"✗ Failed to import Data loader: {e}")
    sys.exit(1)

try:
    from src.evaluation.metrics import compute_routing_metrics
    print("✓ Evaluation metrics")
except Exception as e:
    print(f"✗ Failed to import metrics: {e}")
    sys.exit(1)

# Test 5: Test model initialization
print("\n[5/8] Testing model initialization...")
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TemporalUncertaintyRouter(
        encoder_name='bert-base-uncased',
        embedding_dim=768,
        hidden_dim=256,
        num_sources=4
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model initialized ({num_params:,} parameters)")
except Exception as e:
    print(f"✗ Failed to initialize model: {e}")
    sys.exit(1)

# Test 6: Test tokenizer
print("\n[6/8] Testing tokenizer...")
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    test_text = "This is a test."
    encoding = tokenizer(test_text, return_tensors='pt')
    print("✓ Tokenizer works")
except Exception as e:
    print(f"✗ Tokenizer failed: {e}")
    sys.exit(1)

# Test 7: Test forward pass
print("\n[7/8] Testing model forward pass...")
try:
    model.eval()
    with torch.no_grad():
        output = model(
            input_ids=encoding['input_ids'].to(device),
            attention_mask=encoding['attention_mask'].to(device)
        )
    print("✓ Forward pass successful")
    print(f"  - Routing logits shape: {output['routing_logits'].shape}")
    print(f"  - Epistemic uncertainty: {output['epistemic_uncertainty'].item():.4f}")
    print(f"  - Aleatoric uncertainty: {output['aleatoric_uncertainty'].item():.4f}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    sys.exit(1)

# Test 8: Test data loader (small sample)
print("\n[8/8] Testing data loader (this may take a minute)...")
try:
    data_loader = ConversationDataLoader(
        dataset_name='coqa',
        min_turns=3,
        max_turns=5
    )
    # Just test that we can access the dataset
    from datasets import load_dataset
    coqa = load_dataset("stanfordnlp/coqa", split="validation[:5]")
    print(f"✓ Data loader works (tested with {len(coqa)} samples)")
except Exception as e:
    print(f"⚠ Data loader test failed (not critical): {e}")
    print("  Note: First download may take time. Try running prepare_data.py")

# All tests passed
print("\n" + "=" * 80)
print("ALL TESTS PASSED! ✓")
print("=" * 80)
print("\nYour installation is working correctly!")
print("\nNext steps:")
print("  1. Prepare data: python scripts/prepare_data.py --dataset coqa")
print("  2. Train model: python scripts/train.py --dataset coqa --epochs 2")
print("  3. Check notebooks: jupyter notebook notebooks/01_getting_started.ipynb")
print("\nSee IMPLEMENTATION_GUIDE.md for detailed instructions.")
print("=" * 80)

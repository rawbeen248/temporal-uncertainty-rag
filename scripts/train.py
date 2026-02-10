#!/usr/bin/env python3
"""
Main training script for Temporal Uncertainty Router.

Usage:
    python scripts/train.py --dataset coqa --epochs 20 --batch_size 32
"""

import argparse
import yaml
import torch
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataloader import ConversationDataLoader, create_dataloaders
from src.models.temporal_router import TemporalUncertaintyRouter
from src.training.trainer import Trainer
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Temporal Uncertainty Router')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, default='coqa',
                       choices=['coqa', 'quac', 'all'],
                       help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Data directory')
    parser.add_argument('--cache_dir', type=str, default='./data/cache',
                       help='Cache directory for datasets')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='temporal_router',
                       choices=['temporal_router', 'static_router'],
                       help='Model type')
    parser.add_argument('--encoder_name', type=str, default='bert-base-uncased',
                       help='Pre-trained encoder name')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension')
    parser.add_argument('--num_lstm_layers', type=int, default=2,
                       help='Number of LSTM layers')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                       help='Gradient clipping value')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                       help='Early stopping patience')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--use_amp', action='store_true',
                       help='Use automatic mixed precision')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Output arguments
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='temporal-uncertainty-rag',
                       help='W&B project name')
    
    # Config file
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config YAML file (overrides other args)')
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """Main training function."""
    args = parse_args()
    
    # Load config if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        # Update args with config (args take precedence)
        for key, value in config.items():
            if not hasattr(args, key):
                setattr(args, key, value)
    
    # Set seed
    set_seed(args.seed)
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'
    
    logger.info(f"Using device: {args.device}")
    
    # Create save directory
    save_dir = Path(args.save_dir) / args.dataset / args.model
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Training Configuration")
    logger.info("=" * 80)
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 80)
    
    # Load data
    logger.info(f"\nLoading {args.dataset} dataset...")
    data_loader = ConversationDataLoader(
        dataset_name=args.dataset,
        cache_dir=args.cache_dir,
        min_turns=3,
        max_turns=15
    )
    
    train_conversations, val_conversations = data_loader.load_and_preprocess()
    
    logger.info(f"Loaded {len(train_conversations)} training conversations")
    logger.info(f"Loaded {len(val_conversations)} validation conversations")
    
    # Create dataloaders
    logger.info("\nCreating dataloaders...")
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    
    train_loader, val_loader = create_dataloaders(
        train_conversations=train_conversations,
        val_conversations=val_conversations,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=512,
        include_history=True,
        history_window=5
    )
    
    logger.info(f"Created {len(train_loader)} training batches")
    logger.info(f"Created {len(val_loader)} validation batches")
    
    # Initialize model
    logger.info(f"\nInitializing {args.model}...")
    
    if args.model == 'temporal_router':
        model = TemporalUncertaintyRouter(
            encoder_name=args.encoder_name,
            embedding_dim=768,  # BERT-base
            hidden_dim=args.hidden_dim,
            num_lstm_layers=args.num_lstm_layers,
            num_sources=4,
            dropout=0.1,
            num_mc_samples=10
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {num_params:,} parameters ({num_trainable:,} trainable)")
    
    # Wandb config
    wandb_config = None
    if args.wandb:
        wandb_config = {
            'project': args.wandb_project,
            'name': f'{args.dataset}_{args.model}',
            'config': vars(args)
        }
    
    # Initialize trainer
    logger.info("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        device=args.device,
        checkpoint_dir=str(save_dir),
        use_amp=args.use_amp,
        gradient_clip=args.gradient_clip,
        early_stopping_patience=args.early_stopping_patience,
        wandb_config=wandb_config
    )
    
    # Train
    logger.info("\n" + "=" * 80)
    logger.info("Starting Training")
    logger.info("=" * 80 + "\n")
    
    history = trainer.train()
    
    # Save training history
    import json
    history_path = save_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump({
            'train_losses': [float(x) for x in history['train_losses']],
            'val_metrics': [
                {k: float(v) if isinstance(v, (int, float)) else v 
                 for k, v in metrics.items()}
                for metrics in history['val_metrics']
            ]
        }, f, indent=2)
    
    logger.info(f"\nTraining history saved to {history_path}")
    logger.info(f"Best model saved to {save_dir / 'best_model.pt'}")
    logger.info(f"Best validation F1: {trainer.best_val_metric:.4f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

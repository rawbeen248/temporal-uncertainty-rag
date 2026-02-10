#!/usr/bin/env python3
"""
Data preparation script.
Downloads and preprocesses conversational QA datasets.

Usage:
    python scripts/prepare_data.py --dataset coqa --output_dir data/processed/coqa
    python scripts/prepare_data.py --dataset all --output_dir data/processed
"""

import argparse
import logging
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataloader import ConversationDataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare conversational QA datasets')
    
    parser.add_argument('--dataset', type=str, default='coqa',
                       choices=['coqa', 'quac', 'all'],
                       help='Dataset to prepare (Note: QuAC currently unavailable due to HF deprecation)')
    parser.add_argument('--output_dir', type=str, default='./data/processed',
                       help='Output directory')
    parser.add_argument('--cache_dir', type=str, default='./data/cache',
                       help='Cache directory')
    parser.add_argument('--min_turns', type=int, default=3,
                       help='Minimum turns per conversation')
    parser.add_argument('--max_turns', type=int, default=15,
                       help='Maximum turns per conversation')
    
    return parser.parse_args()


def save_conversations(conversations, output_path):
    """Save conversations to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to serializable format
    data = []
    for conv in conversations:
        conv_data = {
            'conversation_id': conv.conversation_id,
            'context': conv.context,
            'source_dataset': conv.source_dataset,
            'metadata': conv.metadata,
            'turns': [
                {
                    'turn_id': turn.turn_id,
                    'question': turn.question,
                    'answer': turn.answer,
                    'is_answerable': turn.is_answerable,
                    'domain': turn.domain
                }
                for turn in conv.turns
            ]
        }
        data.append(conv_data)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved {len(data)} conversations to {output_path}")


def main():
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("Data Preparation")
    logger.info("=" * 80)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Turn range: {args.min_turns}-{args.max_turns}")
    logger.info("=" * 80 + "\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess data
    data_loader = ConversationDataLoader(
        dataset_name=args.dataset,
        cache_dir=args.cache_dir,
        min_turns=args.min_turns,
        max_turns=args.max_turns
    )
    
    train_conversations, val_conversations = data_loader.load_and_preprocess()
    
    # Save processed data
    if args.dataset == 'all':
        save_conversations(train_conversations, output_dir / 'train_all.json')
        save_conversations(val_conversations, output_dir / 'val_all.json')
    else:
        save_conversations(train_conversations, output_dir / f'train_{args.dataset}.json')
        save_conversations(val_conversations, output_dir / f'val_{args.dataset}.json')
    
    # Print statistics
    logger.info("\n" + "=" * 80)
    logger.info("Data Statistics")
    logger.info("=" * 80)
    logger.info(f"Training conversations: {len(train_conversations)}")
    logger.info(f"Validation conversations: {len(val_conversations)}")
    
    # Turn distribution
    train_turns = [len(conv.turns) for conv in train_conversations]
    val_turns = [len(conv.turns) for conv in val_conversations]
    
    import numpy as np
    logger.info(f"\nTraining turn statistics:")
    logger.info(f"  Mean: {np.mean(train_turns):.2f}")
    logger.info(f"  Std: {np.std(train_turns):.2f}")
    logger.info(f"  Min: {np.min(train_turns)}")
    logger.info(f"  Max: {np.max(train_turns)}")
    
    logger.info(f"\nValidation turn statistics:")
    logger.info(f"  Mean: {np.mean(val_turns):.2f}")
    logger.info(f"  Std: {np.std(val_turns):.2f}")
    logger.info(f"  Min: {np.min(val_turns)}")
    logger.info(f"  Max: {np.max(val_turns)}")
    
    # Domain distribution (for CoQA)
    if args.dataset in ['coqa', 'all']:
        from collections import Counter
        domains = [turn.domain for conv in train_conversations for turn in conv.turns]
        domain_counts = Counter(domains)
        
        logger.info(f"\nDomain distribution:")
        for domain, count in domain_counts.most_common():
            logger.info(f"  {domain}: {count}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Data preparation complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

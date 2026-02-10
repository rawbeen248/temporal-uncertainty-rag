#!/usr/bin/env python3
"""
Merge CoQA and QuAC datasets into a single combined dataset.
"""

import json
import logging
from pathlib import Path
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_json(file_path):
    """Load JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data, file_path):
    """Save data to JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved {len(data)} conversations to {file_path}")


def merge_datasets(coqa_path, quac_path, output_path):
    """Merge CoQA and QuAC datasets."""
    logger.info(f"Loading CoQA from {coqa_path}...")
    coqa_data = load_json(coqa_path)
    
    logger.info(f"Loading QuAC from {quac_path}...")
    quac_data = load_json(quac_path)
    
    # Combine datasets
    merged_data = coqa_data + quac_data
    
    logger.info(f"Merged dataset statistics:")
    logger.info(f"  CoQA conversations: {len(coqa_data)}")
    logger.info(f"  QuAC conversations: {len(quac_data)}")
    logger.info(f"  Total conversations: {len(merged_data)}")
    
    # Save merged dataset
    save_json(merged_data, output_path)
    
    return merged_data


def main():
    parser = argparse.ArgumentParser(description='Merge CoQA and QuAC datasets')
    parser.add_argument('--coqa_dir', type=str, default='data/processed/coqa',
                       help='Directory containing CoQA processed files')
    parser.add_argument('--quac_dir', type=str, default='data/processed/quac',
                       help='Directory containing QuAC processed files')
    parser.add_argument('--output_dir', type=str, default='data/processed/combined',
                       help='Output directory for merged files')
    args = parser.parse_args()
    
    coqa_dir = Path(args.coqa_dir)
    quac_dir = Path(args.quac_dir)
    output_dir = Path(args.output_dir)
    
    # Merge training sets
    logger.info("\n" + "="*80)
    logger.info("Merging Training Sets")
    logger.info("="*80)
    merge_datasets(
        coqa_dir / 'train_coqa.json',
        quac_dir / 'train_quac.json',
        output_dir / 'train_combined.json'
    )
    
    # Merge validation sets
    logger.info("\n" + "="*80)
    logger.info("Merging Validation Sets")
    logger.info("="*80)
    merge_datasets(
        coqa_dir / 'val_coqa.json',
        quac_dir / 'val_quac.json',
        output_dir / 'val_combined.json'
    )
    
    logger.info("\n" + "="*80)
    logger.info("Dataset Merging Complete!")
    logger.info("="*80)
    logger.info(f"Combined datasets saved to: {output_dir}")
    logger.info("\nYou can now train using:")
    logger.info(f"  python scripts/train.py --train_data {output_dir}/train_combined.json --val_data {output_dir}/val_combined.json")


if __name__ == '__main__':
    main()

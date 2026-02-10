#!/usr/bin/env python3
"""
Manual QuAC dataset downloader and processor.
Downloads QuAC from the original source and converts to our format.
"""

import json
import logging
import argparse
from pathlib import Path
import urllib.request
from typing import List, Dict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.dataloader import Conversation, ConversationTurn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# QuAC dataset URLs (from official GitHub)
QUAC_URLS = {
    'train': 'https://s3.amazonaws.com/my89public/quac/train_v0.2.json',
    'val': 'https://s3.amazonaws.com/my89public/quac/val_v0.2.json'
}


def download_file(url: str, output_path: Path):
    """Download file from URL."""
    logger.info(f"Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, output_path)
        logger.info(f"Downloaded to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def process_quac_data(data: Dict, split_name: str, min_turns: int, max_turns: int) -> List[Conversation]:
    """Process QuAC JSON data into Conversation objects."""
    conversations = []
    
    for article_idx, article in enumerate(data['data']):
        title = article.get('title', 'unknown')
        section_title = article.get('section_title', 'unknown')
        
        for para_idx, paragraph in enumerate(article['paragraphs']):
            context = paragraph['context']
            qas = paragraph['qas']
            
            # Each paragraph's QA list forms one multi-turn conversation
            turns = []
            
            for turn_id, qa in enumerate(qas):
                question = qa.get('question', '')
                answers = qa.get('answers', [])
                
                if not answers:
                    # Use orig_answer as fallback
                    orig_answer = qa.get('orig_answer', {})
                    answer_text = orig_answer.get('text', 'CANNOTANSWER')
                    answer_start = orig_answer.get('answer_start', None)
                else:
                    answer_text = answers[0].get('text', 'CANNOTANSWER')
                    answer_start = answers[0].get('answer_start', None)
                
                is_answerable = answer_text != 'CANNOTANSWER'
                
                turn = ConversationTurn(
                    turn_id=turn_id,
                    question=question,
                    answer=answer_text,
                    is_answerable=is_answerable,
                    domain='wikipedia',
                    answer_start=answer_start
                )
                turns.append(turn)
            
            # Filter by turn count
            if min_turns <= len(turns) <= max_turns:
                conversation = Conversation(
                    conversation_id=f"quac_{split_name}_{article_idx}_{para_idx}",
                    context=context,
                    turns=turns,
                    source_dataset='quac',
                    metadata={
                        'title': title,
                        'section_title': section_title,
                        'paragraph_id': paragraph.get('id', 'unknown')
                    }
                )
                conversations.append(conversation)
    
    return conversations


def main():
    parser = argparse.ArgumentParser(description='Download and process QuAC dataset manually')
    parser.add_argument('--output_dir', type=str, default='data/quac_raw',
                       help='Directory to save downloaded files')
    parser.add_argument('--processed_dir', type=str, default='data/processed/quac',
                       help='Directory to save processed files')
    parser.add_argument('--min_turns', type=int, default=3)
    parser.add_argument('--max_turns', type=int, default=15)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    processed_dir = Path(args.processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Download datasets
    for split, url in QUAC_URLS.items():
        output_path = output_dir / f'{split}_v0.2.json'
        
        if not output_path.exists():
            success = download_file(url, output_path)
            if not success:
                logger.error(f"Failed to download {split} split")
                continue
        else:
            logger.info(f"{split} split already downloaded")
        
        # Load and process
        logger.info(f"Processing {split} split...")
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        conversations = process_quac_data(data, split, args.min_turns, args.max_turns)
        logger.info(f"Processed {len(conversations)} conversations from {split}")
        
        # Save processed data
        output_file = processed_dir / f'{split}_quac.json'
        conv_data = []
        for conv in conversations:
            conv_data.append({
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
            })
        
        with open(output_file, 'w') as f:
            json.dump(conv_data, f, indent=2)
        logger.info(f"Saved to {output_file}")
    
    logger.info("\n" + "="*80)
    logger.info("QuAC manual download and processing complete!")
    logger.info("Note: The QuAC format is different from CoQA. You may need to adjust")
    logger.info("the conversation reconstruction logic for better multi-turn support.")
    logger.info("="*80)


if __name__ == '__main__':
    main()

"""
Data loading and preprocessing for conversational QA datasets.
Handles CoQA and QuAC datasets from HuggingFace.
"""

import os
import json
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation."""
    turn_id: int
    question: str
    answer: str
    is_answerable: bool
    domain: Optional[str] = None
    answer_start: Optional[int] = None
    answer_end: Optional[int] = None


@dataclass
class Conversation:
    """Represents a complete conversation."""
    conversation_id: str
    context: str
    turns: List[ConversationTurn]
    source_dataset: str
    metadata: Optional[Dict] = None


class ConversationDataLoader:
    """
    Loads and preprocesses conversational QA datasets (CoQA, QuAC).
    
    Args:
        dataset_name: Name of dataset ('coqa', 'quac', or 'all')
        cache_dir: Directory to cache downloaded datasets
        min_turns: Minimum number of turns per conversation
        max_turns: Maximum number of turns per conversation
    """
    
    def __init__(
        self,
        dataset_name: str = 'coqa',
        cache_dir: str = './data/cache',
        min_turns: int = 3,
        max_turns: int = 15
    ):
        self.dataset_name = dataset_name.lower()
        self.cache_dir = cache_dir
        self.min_turns = min_turns
        self.max_turns = max_turns
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        logger.info(f"Initializing ConversationDataLoader for {dataset_name}")
    
    def load_and_preprocess(self) -> Tuple[List[Conversation], List[Conversation]]:
        """
        Load and preprocess datasets.
        
        Returns:
            Tuple of (train_conversations, validation_conversations)
        """
        if self.dataset_name == 'coqa':
            return self._load_coqa()
        elif self.dataset_name == 'quac':
            return self._load_quac()
        elif self.dataset_name == 'all':
            coqa_train, coqa_val = self._load_coqa()
            quac_train, quac_val = self._load_quac()
            return coqa_train + quac_train, coqa_val + quac_val
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _load_coqa(self) -> Tuple[List[Conversation], List[Conversation]]:
        """Load and preprocess CoQA dataset."""
        logger.info("Loading CoQA dataset from HuggingFace...")
        
        # Load dataset
        dataset = load_dataset("stanfordnlp/coqa", cache_dir=self.cache_dir)
        
        # Process train and validation splits
        train_conversations = self._process_coqa_split(
            dataset['train'], 
            split_name='train'
        )
        val_conversations = self._process_coqa_split(
            dataset['validation'], 
            split_name='validation'
        )
        
        logger.info(f"CoQA: {len(train_conversations)} train, {len(val_conversations)} val conversations")
        
        return train_conversations, val_conversations
    
    def _process_coqa_split(self, split_data, split_name: str) -> List[Conversation]:
        """Process a CoQA data split."""
        conversations = []
        
        for idx, example in enumerate(split_data):
            try:
                # Extract fields
                story = example['story']
                questions = example['questions']
                answers = example['answers']
                source = example.get('source', 'unknown')
                
                # Create turns
                turns = []
                num_turns = min(len(questions), len(answers['input_text']))
                
                for turn_id in range(num_turns):
                    turn = ConversationTurn(
                        turn_id=turn_id,
                        question=questions[turn_id],
                        answer=answers['input_text'][turn_id],
                        is_answerable=True,  # CoQA doesn't have explicit unanswerable
                        domain=source,
                        answer_start=answers['answer_start'][turn_id] if 'answer_start' in answers else None,
                        answer_end=answers['answer_end'][turn_id] if 'answer_end' in answers else None
                    )
                    turns.append(turn)
                
                # Filter by turn count
                if self.min_turns <= len(turns) <= self.max_turns:
                    conversation = Conversation(
                        conversation_id=f"coqa_{split_name}_{idx}",
                        context=story,
                        turns=turns,
                        source_dataset='coqa',
                        metadata={'source': source}
                    )
                    conversations.append(conversation)
            
            except Exception as e:
                logger.warning(f"Error processing CoQA example {idx}: {e}")
                continue
        
        return conversations
    
    def _load_quac(self) -> Tuple[List[Conversation], List[Conversation]]:
        """Load and preprocess QuAC dataset."""
        logger.warning("QuAC dataset loading is currently disabled due to HuggingFace deprecating dataset loading scripts.")
        logger.warning("Returning empty conversation lists. Please use CoQA dataset instead.")
        logger.info("To use QuAC, the dataset needs to be converted to Parquet format by the dataset maintainer.")
        
        # Return empty lists as QuAC is currently unsupported
        return [], []
        
        # NOTE: The following code is disabled because HuggingFace no longer supports dataset loading scripts
        # logger.info("Loading QuAC dataset from HuggingFace...")
        # dataset = load_dataset("allenai/quac", cache_dir=self.cache_dir, trust_remote_code=True)
        
        # Process train and validation splits
        train_conversations = self._process_quac_split(
            dataset['train'], 
            split_name='train'
        )
        val_conversations = self._process_quac_split(
            dataset['validation'], 
            split_name='validation'
        )
        
        logger.info(f"QuAC: {len(train_conversations)} train, {len(val_conversations)} val conversations")
        
        return train_conversations, val_conversations
    
    def _process_quac_split(self, split_data, split_name: str) -> List[Conversation]:
        """Process a QuAC data split."""
        conversations = []
        
        for idx, example in enumerate(split_data):
            try:
                # Extract fields
                context = example['context']
                questions = example['questions']
                answers = example['answers']
                
                # Create turns
                turns = []
                num_turns = min(len(questions), len(answers['texts']))
                
                for turn_id in range(num_turns):
                    # Check if answerable
                    answer_text = answers['texts'][turn_id][0] if answers['texts'][turn_id] else "CANNOTANSWER"
                    is_answerable = answer_text != "CANNOTANSWER"
                    
                    turn = ConversationTurn(
                        turn_id=turn_id,
                        question=questions[turn_id],
                        answer=answer_text,
                        is_answerable=is_answerable,
                        domain='wikipedia',
                        answer_start=answers['answer_starts'][turn_id][0] if answers['answer_starts'][turn_id] else None
                    )
                    turns.append(turn)
                
                # Filter by turn count
                if self.min_turns <= len(turns) <= self.max_turns:
                    conversation = Conversation(
                        conversation_id=f"quac_{split_name}_{idx}",
                        context=context,
                        turns=turns,
                        source_dataset='quac',
                        metadata={'section_title': example.get('section_title', 'unknown')}
                    )
                    conversations.append(conversation)
            
            except Exception as e:
                logger.warning(f"Error processing QuAC example {idx}: {e}")
                continue
        
        return conversations


class ConversationalDataset(Dataset):
    """
    PyTorch Dataset for conversational QA with temporal features.
    
    Each sample represents a conversation turn with its history.
    """
    
    def __init__(
        self,
        conversations: List[Conversation],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        include_history: bool = True,
        history_window: int = 5
    ):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_history = include_history
        self.history_window = history_window
        
        # Flatten conversations into individual turns with history
        self.samples = self._create_samples()
        
        logger.info(f"Created {len(self.samples)} training samples from {len(conversations)} conversations")
    
    def _create_samples(self) -> List[Dict]:
        """Create individual training samples from conversations."""
        samples = []
        
        for conv in self.conversations:
            for turn_idx, turn in enumerate(conv.turns):
                # Get conversation history
                history = conv.turns[:turn_idx] if turn_idx > 0 else []
                
                # Limit history window
                if len(history) > self.history_window:
                    history = history[-self.history_window:]
                
                sample = {
                    'conversation_id': conv.conversation_id,
                    'turn_id': turn.turn_id,
                    'context': conv.context,
                    'current_question': turn.question,
                    'current_answer': turn.answer,
                    'is_answerable': turn.is_answerable,
                    'history': history,
                    'domain': turn.domain,
                    'source_dataset': conv.source_dataset
                }
                
                samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single training sample."""
        sample = self.samples[idx]
        
        # Format input text with history
        input_text = self._format_input(sample)
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create labels (ground truth routing decision)
        # This is a simplified version - you'll need to implement actual routing logic
        routing_label = self._get_routing_label(sample)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'routing_label': torch.tensor(routing_label, dtype=torch.long),
            'is_answerable': torch.tensor(sample['is_answerable'], dtype=torch.float),
            'turn_id': torch.tensor(sample['turn_id'], dtype=torch.long),
            'conversation_id': sample['conversation_id']
        }
    
    def _format_input(self, sample: Dict) -> str:
        """Format input text with conversation history."""
        parts = []
        
        # Add context
        parts.append(f"Context: {sample['context']}")
        
        # Add conversation history
        if self.include_history and sample['history']:
            parts.append("Conversation History:")
            for turn in sample['history']:
                parts.append(f"Q: {turn.question}")
                parts.append(f"A: {turn.answer}")
        
        # Add current question
        parts.append(f"Current Question: {sample['current_question']}")
        
        return " ".join(parts)
    
    def _get_routing_label(self, sample: Dict) -> int:
        """
        Determine ground truth routing label.
        
        Routing options:
        0: Internal KB
        1: External Search
        2: Clarification Question
        3: Multi-source Fusion
        
        This is a heuristic - you may want to refine this logic.
        """
        # Simple heuristic based on answerability and turn position
        if not sample['is_answerable']:
            return 2  # Clarification needed
        elif sample['turn_id'] == 0:
            return 1  # External search for first turn
        elif len(sample['history']) >= 3:
            return 3  # Multi-source fusion for later turns
        else:
            return 0  # Internal KB for middle turns
        
        # Note: In actual implementation, you'd want more sophisticated labeling
        # based on answer source, query complexity, etc.


def create_dataloaders(
    train_conversations: List[Conversation],
    val_conversations: List[Conversation],
    tokenizer: AutoTokenizer,
    batch_size: int = 32,
    num_workers: int = 4,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training and validation.
    
    Args:
        train_conversations: Training conversations
        val_conversations: Validation conversations
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size
        num_workers: Number of data loading workers
        **dataset_kwargs: Additional arguments for ConversationalDataset
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = ConversationalDataset(train_conversations, tokenizer, **dataset_kwargs)
    val_dataset = ConversationalDataset(val_conversations, tokenizer, **dataset_kwargs)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Example usage
    from transformers import AutoTokenizer
    
    # Load data
    data_loader = ConversationDataLoader(dataset_name='coqa')
    train_convs, val_convs = data_loader.load_and_preprocess()
    
    print(f"Loaded {len(train_convs)} training conversations")
    print(f"Loaded {len(val_convs)} validation conversations")
    
    # Example conversation
    if train_convs:
        conv = train_convs[0]
        print(f"\nExample conversation: {conv.conversation_id}")
        print(f"Context: {conv.context[:200]}...")
        print(f"Number of turns: {len(conv.turns)}")
        for turn in conv.turns[:3]:
            print(f"\nTurn {turn.turn_id}:")
            print(f"  Q: {turn.question}")
            print(f"  A: {turn.answer}")
    
    # Create dataset
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    train_loader, val_loader = create_dataloaders(
        train_convs, val_convs, tokenizer, batch_size=8
    )
    
    print(f"\nCreated DataLoaders:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Show sample batch
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")

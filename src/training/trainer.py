"""
Training module for Temporal Uncertainty Router.
Handles training loop, validation, checkpointing, and early stopping.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional, List
import logging
from tqdm import tqdm
import numpy as np
from pathlib import Path

from ..models.temporal_router import TemporalUncertaintyRouter, ConversationState
from ..models.uncertainty_estimator import UncertaintyLoss
from ..evaluation.metrics import compute_routing_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for Temporal Uncertainty Router.
    
    Args:
        model: TemporalUncertaintyRouter model
        train_loader: Training data loader
        val_loader: Validation data loader
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        device: Training device ('cuda' or 'cpu')
        checkpoint_dir: Directory to save checkpoints
        use_amp: Whether to use automatic mixed precision
        gradient_clip: Gradient clipping value
        early_stopping_patience: Patience for early stopping
        wandb_config: Optional wandb configuration
    """
    
    def __init__(
        self,
        model: TemporalUncertaintyRouter,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 1e-4,
        num_epochs: int = 20,
        device: str = 'cuda',
        checkpoint_dir: str = './checkpoints',
        use_amp: bool = True,
        gradient_clip: float = 1.0,
        early_stopping_patience: int = 5,
        wandb_config: Optional[Dict] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.use_amp = use_amp and torch.cuda.is_available()
        self.gradient_clip = gradient_clip
        self.early_stopping_patience = early_stopping_patience
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Loss functions
        self.routing_loss_fn = nn.CrossEntropyLoss()
        self.uncertainty_loss_fn = UncertaintyLoss(
            calibration_weight=1.0,
            reg_weight=0.1
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.current_epoch = 0
        self.best_val_metric = 0.0
        self.patience_counter = 0
        self.train_losses = []
        self.val_metrics = []
        
        # Wandb logging
        self.use_wandb = wandb_config is not None
        if self.use_wandb:
            import wandb
            wandb.init(**wandb_config)
            wandb.watch(model)
        
        logger.info(f"Trainer initialized. Device: {device}, AMP: {self.use_amp}")
    
    def train(self) -> Dict[str, List[float]]:
        """
        Main training loop.
        
        Returns:
            Dictionary with training history
        """
        logger.info(f"Starting training for {self.num_epochs} epochs...")
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            # Train one epoch
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_metrics = self.validate()
            self.val_metrics.append(val_metrics)
            
            # Log
            self._log_epoch(train_loss, val_metrics)
            
            # Save checkpoint
            self._save_checkpoint(val_metrics)
            
            # Early stopping check
            if self._check_early_stopping(val_metrics):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        logger.info("Training completed!")
        
        return {
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics
        }
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    loss = self._compute_loss(batch)
            else:
                loss = self._compute_loss(batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()
            
            # Update scheduler
            self.scheduler.step()
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def _compute_loss(self, batch: Dict) -> torch.Tensor:
        """
        Compute combined loss for a batch.
        
        Loss = routing_loss + uncertainty_loss + consistency_loss
        """
        # Get model outputs
        # Note: This is simplified - you need to handle conversation history properly
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        
        # Routing loss (cross-entropy)
        routing_loss = self.routing_loss_fn(
            output['routing_logits'],
            batch['routing_label']
        )
        
        # Uncertainty loss (calibration)
        uncertainty_loss = self.uncertainty_loss_fn(
            predictions=output['routing_logits'],
            targets=batch['routing_label'],
            epistemic=output['epistemic_uncertainty'],
            aleatoric=output['aleatoric_uncertainty']
        )
        
        # Consistency loss (encourage stable routing for similar queries)
        # This is optional and can be refined
        consistency_loss = 0.0
        
        # Combined loss
        total_loss = (
            1.0 * routing_loss +
            0.5 * uncertainty_loss +
            0.3 * consistency_loss
        )
        
        return total_loss
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate on validation set.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_uncertainties = {'epistemic': [], 'aleatoric': []}
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            output = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            # Collect predictions
            predictions = torch.argmax(output['routing_logits'], dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch['routing_label'].cpu().numpy())
            
            # Collect uncertainties
            all_uncertainties['epistemic'].extend(
                output['epistemic_uncertainty'].cpu().numpy()
            )
            all_uncertainties['aleatoric'].extend(
                output['aleatoric_uncertainty'].cpu().numpy()
            )
        
        # Compute metrics
        metrics = compute_routing_metrics(
            predictions=np.array(all_predictions),
            targets=np.array(all_targets)
        )
        
        # Add uncertainty statistics
        metrics['mean_epistemic'] = np.mean(all_uncertainties['epistemic'])
        metrics['mean_aleatoric'] = np.mean(all_uncertainties['aleatoric'])
        
        return metrics
    
    def _log_epoch(self, train_loss: float, val_metrics: Dict[str, float]):
        """Log epoch results."""
        log_str = f"Epoch {self.current_epoch}: "
        log_str += f"Train Loss: {train_loss:.4f}, "
        log_str += f"Val Acc: {val_metrics['accuracy']:.4f}, "
        log_str += f"Val F1: {val_metrics['f1']:.4f}"
        
        logger.info(log_str)
        
        # Wandb logging
        if self.use_wandb:
            import wandb
            wandb.log({
                'epoch': self.current_epoch,
                'train_loss': train_loss,
                **{f'val_{k}': v for k, v in val_metrics.items()}
            })
    
    def _save_checkpoint(self, val_metrics: Dict[str, float]):
        """Save model checkpoint."""
        # Save latest
        latest_path = self.checkpoint_dir / 'latest.pt'
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics,
        }, latest_path)
        
        # Save best
        current_metric = val_metrics['f1']
        if current_metric > self.best_val_metric:
            self.best_val_metric = current_metric
            self.patience_counter = 0
            
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'val_metrics': val_metrics,
                'best_val_f1': self.best_val_metric
            }, best_path)
            
            logger.info(f"New best model saved! F1: {self.best_val_metric:.4f}")
    
    def _check_early_stopping(self, val_metrics: Dict[str, float]) -> bool:
        """Check if early stopping should be triggered."""
        current_metric = val_metrics['f1']
        
        if current_metric <= self.best_val_metric:
            self.patience_counter += 1
        
        if self.patience_counter >= self.early_stopping_patience:
            return True
        
        return False
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'epoch' in checkpoint:
            self.current_epoch = checkpoint['epoch']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")


class MultiTaskTrainer(Trainer):
    """
    Extended trainer for multi-task learning.
    
    Trains routing + answer generation + uncertainty estimation jointly.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Additional loss function for answer generation
        self.answer_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    def _compute_loss(self, batch: Dict) -> torch.Tensor:
        """Compute multi-task loss."""
        # Get routing loss (from parent class)
        base_loss = super()._compute_loss(batch)
        
        # If batch contains answer generation targets, add answer loss
        if 'answer_labels' in batch:
            # This is a placeholder - implement answer generation head
            # answer_logits = self.model.generate_answer(...)
            # answer_loss = self.answer_loss_fn(answer_logits, batch['answer_labels'])
            # return base_loss + 0.5 * answer_loss
            pass
        
        return base_loss


if __name__ == "__main__":
    # Example usage
    from transformers import AutoTokenizer
    from ..models.temporal_router import TemporalUncertaintyRouter
    from ..data.dataloader import ConversationDataLoader, create_dataloaders
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load data
    data_loader = ConversationDataLoader(dataset_name='coqa')
    train_convs, val_convs = data_loader.load_and_preprocess()
    
    # Create dataloaders
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    train_loader, val_loader = create_dataloaders(
        train_convs[:100],  # Use subset for testing
        val_convs[:20],
        tokenizer,
        batch_size=8
    )
    
    # Initialize model
    model = TemporalUncertaintyRouter(
        encoder_name='bert-base-uncased',
        embedding_dim=768,
        hidden_dim=256,
        num_sources=4
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-4,
        num_epochs=3,
        device=device,
        checkpoint_dir='./checkpoints/test'
    )
    
    # Train
    history = trainer.train()
    
    print(f"\nTraining completed!")
    print(f"Final train loss: {history['train_losses'][-1]:.4f}")
    print(f"Best val F1: {trainer.best_val_metric:.4f}")

"""
Uncertainty estimation module for conversational QA.
Implements epistemic and aleatoric uncertainty estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class UncertaintyEstimator(nn.Module):
    """
    Estimates epistemic and aleatoric uncertainty for conversational QA.
    
    Epistemic uncertainty (model uncertainty): Uncertainty due to lack of knowledge
    Aleatoric uncertainty (data uncertainty): Inherent ambiguity in the query
    
    Args:
        input_dim: Dimension of input features
        hidden_dim: Hidden layer dimension
        num_mc_samples: Number of MC dropout samples for epistemic uncertainty
        dropout_rate: Dropout rate for MC dropout
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_mc_samples: int = 10,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_mc_samples = num_mc_samples
        self.dropout_rate = dropout_rate
        
        # Shared feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Epistemic uncertainty estimation (via MC dropout)
        self.epistemic_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Normalize to [0, 1]
        )
        
        # Aleatoric uncertainty estimation (learned)
        self.aleatoric_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensure positive values
        )
    
    def forward(
        self, 
        features: torch.Tensor, 
        return_samples: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass to estimate uncertainties.
        
        Args:
            features: Input features [batch_size, input_dim]
            return_samples: Whether to return MC samples
        
        Returns:
            epistemic_uncertainty: [batch_size, 1]
            aleatoric_uncertainty: [batch_size, 1]
            mc_samples: Optional[batch_size, num_mc_samples] if return_samples=True
        """
        batch_size = features.size(0)
        
        # Extract shared features
        shared_features = self.feature_extractor(features)
        
        # Estimate aleatoric uncertainty (single forward pass)
        aleatoric = self.aleatoric_head(shared_features)
        
        # Estimate epistemic uncertainty (MC dropout)
        epistemic, mc_samples = self._estimate_epistemic(
            shared_features, 
            return_samples=return_samples
        )
        
        return epistemic, aleatoric, mc_samples
    
    def _estimate_epistemic(
        self, 
        features: torch.Tensor,
        return_samples: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Estimate epistemic uncertainty using MC dropout.
        
        Args:
            features: Shared features [batch_size, hidden_dim]
            return_samples: Whether to return individual MC samples
        
        Returns:
            epistemic_uncertainty: Variance across MC samples [batch_size, 1]
            mc_samples: Optional[batch_size, num_mc_samples]
        """
        batch_size = features.size(0)
        
        # Enable dropout during inference
        self.epistemic_head.train()
        
        # Collect MC samples
        samples = []
        for _ in range(self.num_mc_samples):
            sample = self.epistemic_head(features)
            samples.append(sample)
        
        # Stack samples [num_mc_samples, batch_size, 1]
        samples = torch.stack(samples, dim=0)
        
        # Compute variance as epistemic uncertainty
        epistemic = torch.var(samples, dim=0)
        
        # Return to eval mode
        self.epistemic_head.eval()
        
        if return_samples:
            return epistemic, samples.squeeze(-1).transpose(0, 1)  # [batch_size, num_mc_samples]
        else:
            return epistemic, None
    
    def compute_total_uncertainty(
        self, 
        epistemic: torch.Tensor, 
        aleatoric: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine epistemic and aleatoric uncertainties.
        
        Total uncertainty = sqrt(epistemic^2 + aleatoric^2)
        
        Args:
            epistemic: Epistemic uncertainty [batch_size, 1]
            aleatoric: Aleatoric uncertainty [batch_size, 1]
        
        Returns:
            total_uncertainty: [batch_size, 1]
        """
        return torch.sqrt(epistemic ** 2 + aleatoric ** 2 + 1e-8)


class TemporalUncertaintyTracker(nn.Module):
    """
    Tracks uncertainty evolution across conversation turns.
    
    Computes temporal metrics:
    - Uncertainty Decay Rate (UDR)
    - Epistemic Convergence Speed (ECS)
    
    Args:
        window_size: Number of previous turns to track
    """
    
    def __init__(self, window_size: int = 5):
        super().__init__()
        self.window_size = window_size
    
    def forward(
        self,
        current_epistemic: torch.Tensor,
        current_aleatoric: torch.Tensor,
        history_epistemic: Optional[torch.Tensor] = None,
        history_aleatoric: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Compute temporal uncertainty metrics.
        
        Args:
            current_epistemic: Current turn epistemic uncertainty [batch_size, 1]
            current_aleatoric: Current turn aleatoric uncertainty [batch_size, 1]
            history_epistemic: Previous turns epistemic [batch_size, num_prev_turns]
            history_aleatoric: Previous turns aleatoric [batch_size, num_prev_turns]
        
        Returns:
            Dictionary with temporal metrics
        """
        metrics = {}
        
        if history_epistemic is not None and history_epistemic.size(1) > 0:
            # Compute Uncertainty Decay Rate (UDR)
            metrics['udr'] = self.compute_udr(
                current_epistemic, 
                current_aleatoric,
                history_epistemic, 
                history_aleatoric
            )
            
            # Compute Epistemic Convergence Speed (ECS)
            metrics['ecs'] = self.compute_ecs(
                current_epistemic,
                history_epistemic
            )
            
            # Compute uncertainty trend
            metrics['epistemic_trend'] = self.compute_trend(history_epistemic)
            metrics['aleatoric_trend'] = self.compute_trend(history_aleatoric)
        else:
            # First turn - no history
            metrics['udr'] = torch.zeros_like(current_epistemic)
            metrics['ecs'] = torch.zeros_like(current_epistemic)
            metrics['epistemic_trend'] = torch.zeros_like(current_epistemic)
            metrics['aleatoric_trend'] = torch.zeros_like(current_aleatoric)
        
        return metrics
    
    def compute_udr(
        self,
        current_epistemic: torch.Tensor,
        current_aleatoric: torch.Tensor,
        history_epistemic: torch.Tensor,
        history_aleatoric: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Uncertainty Decay Rate (UDR).
        
        UDR = (U_prev - U_current) / max(U_prev, epsilon)
        where U is total uncertainty
        
        Args:
            current_epistemic: [batch_size, 1]
            current_aleatoric: [batch_size, 1]
            history_epistemic: [batch_size, num_prev_turns]
            history_aleatoric: [batch_size, num_prev_turns]
        
        Returns:
            udr: [batch_size, 1]
        """
        # Compute total uncertainties
        current_total = torch.sqrt(current_epistemic ** 2 + current_aleatoric ** 2)
        
        # Get most recent historical total uncertainty
        prev_epistemic = history_epistemic[:, -1:]  # [batch_size, 1]
        prev_aleatoric = history_aleatoric[:, -1:]
        prev_total = torch.sqrt(prev_epistemic ** 2 + prev_aleatoric ** 2)
        
        # Compute decay rate
        udr = (prev_total - current_total) / (prev_total + 1e-8)
        
        return udr
    
    def compute_ecs(
        self,
        current_epistemic: torch.Tensor,
        history_epistemic: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Epistemic Convergence Speed (ECS).
        
        ECS measures how quickly epistemic uncertainty is converging to zero.
        Higher values indicate faster convergence.
        
        Args:
            current_epistemic: [batch_size, 1]
            history_epistemic: [batch_size, num_prev_turns]
        
        Returns:
            ecs: [batch_size, 1]
        """
        if history_epistemic.size(1) < 2:
            return torch.zeros_like(current_epistemic)
        
        # Compute slope of epistemic uncertainty over time
        # Using linear regression: slope = cov(x,y) / var(x)
        num_turns = history_epistemic.size(1)
        
        # Time indices
        x = torch.arange(num_turns, dtype=torch.float32, device=history_epistemic.device)
        x = x.unsqueeze(0).expand(history_epistemic.size(0), -1)  # [batch_size, num_turns]
        
        # Compute means
        x_mean = x.mean(dim=1, keepdim=True)
        y_mean = history_epistemic.mean(dim=1, keepdim=True)
        
        # Compute covariance and variance
        cov = ((x - x_mean) * (history_epistemic - y_mean)).sum(dim=1, keepdim=True)
        var = ((x - x_mean) ** 2).sum(dim=1, keepdim=True)
        
        # Slope (negative slope = convergence)
        slope = cov / (var + 1e-8)
        
        # ECS is negative slope (higher = faster convergence)
        ecs = -slope
        
        return ecs
    
    def compute_trend(self, history: torch.Tensor) -> torch.Tensor:
        """
        Compute trend direction (increasing/decreasing/stable).
        
        Args:
            history: [batch_size, num_prev_turns]
        
        Returns:
            trend: [batch_size, 1] (-1: decreasing, 0: stable, 1: increasing)
        """
        if history.size(1) < 2:
            return torch.zeros(history.size(0), 1, device=history.device)
        
        # Compare recent average to earlier average
        mid_point = history.size(1) // 2
        recent_avg = history[:, mid_point:].mean(dim=1, keepdim=True)
        earlier_avg = history[:, :mid_point].mean(dim=1, keepdim=True)
        
        # Compute trend
        diff = recent_avg - earlier_avg
        trend = torch.sign(diff)
        
        return trend


class UncertaintyLoss(nn.Module):
    """
    Custom loss function for uncertainty estimation.
    
    Combines:
    1. Calibration loss (uncertainty should match prediction error)
    2. Regularization loss (prevent extreme uncertainty values)
    """
    
    def __init__(self, calibration_weight: float = 1.0, reg_weight: float = 0.1):
        super().__init__()
        self.calibration_weight = calibration_weight
        self.reg_weight = reg_weight
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        epistemic: torch.Tensor,
        aleatoric: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute uncertainty loss.
        
        Args:
            predictions: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            epistemic: Epistemic uncertainty [batch_size, 1]
            aleatoric: Aleatoric uncertainty [batch_size, 1]
        
        Returns:
            loss: Scalar loss value
        """
        # Prediction error
        pred_error = F.cross_entropy(predictions, targets, reduction='none')
        
        # Total uncertainty
        total_uncertainty = torch.sqrt(epistemic.squeeze() ** 2 + aleatoric.squeeze() ** 2)
        
        # Calibration loss: uncertainty should correlate with error
        calibration_loss = F.mse_loss(total_uncertainty, pred_error)
        
        # Regularization: prevent extreme values
        reg_loss = torch.mean(epistemic) + torch.mean(aleatoric)
        
        # Combined loss
        total_loss = (
            self.calibration_weight * calibration_loss +
            self.reg_weight * reg_loss
        )
        
        return total_loss


if __name__ == "__main__":
    # Example usage
    batch_size = 4
    input_dim = 768
    
    # Create estimator
    estimator = UncertaintyEstimator(
        input_dim=input_dim,
        hidden_dim=256,
        num_mc_samples=10
    )
    
    # Random features
    features = torch.randn(batch_size, input_dim)
    
    # Estimate uncertainties
    epistemic, aleatoric, samples = estimator(features, return_samples=True)
    
    print(f"Epistemic uncertainty shape: {epistemic.shape}")
    print(f"Aleatoric uncertainty shape: {aleatoric.shape}")
    print(f"MC samples shape: {samples.shape if samples is not None else None}")
    
    print(f"\nEpistemic values: {epistemic.squeeze().detach().numpy()}")
    print(f"Aleatoric values: {aleatoric.squeeze().detach().numpy()}")
    
    # Test temporal tracker
    tracker = TemporalUncertaintyTracker(window_size=5)
    
    # Simulate history
    history_epistemic = torch.randn(batch_size, 3).abs()
    history_aleatoric = torch.randn(batch_size, 3).abs()
    
    metrics = tracker(epistemic, aleatoric, history_epistemic, history_aleatoric)
    
    print(f"\nTemporal metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value.squeeze().detach().numpy()}")

"""
Evaluation metrics for Temporal Uncertainty Tracking in Conversational RAG.

Implements:
1. Standard metrics (accuracy, F1, precision, recall)
2. Novel temporal metrics (UDR, ECS, RAS)
3. Statistical significance testing
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from scipy import stats
from dataclasses import dataclass


@dataclass
class TemporalMetrics:
    """Container for temporal uncertainty metrics."""
    uncertainty_decay_rate: float  # UDR
    epistemic_convergence_speed: float  # ECS
    routing_adaptation_score: float  # RAS
    epistemic_std: float
    aleatoric_std: float
    total_uncertainty_mean: float


def compute_routing_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    average: str = 'macro'
) -> Dict[str, float]:
    """
    Compute standard routing performance metrics.
    
    Args:
        predictions: Predicted routing decisions [num_samples]
        targets: Ground truth routing decisions [num_samples]
        average: Averaging method for multi-class metrics
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(targets, predictions),
        'f1': f1_score(targets, predictions, average=average, zero_division=0),
        'precision': precision_score(targets, predictions, average=average, zero_division=0),
        'recall': recall_score(targets, predictions, average=average, zero_division=0)
    }
    
    # Per-class metrics
    f1_per_class = f1_score(targets, predictions, average=None, zero_division=0)
    for i, f1_val in enumerate(f1_per_class):
        metrics[f'f1_class_{i}'] = f1_val
    
    return metrics


def compute_uncertainty_decay_rate(
    uncertainty_history: List[float],
    window_size: int = 3
) -> float:
    """
    Compute Uncertainty Decay Rate (UDR).
    
    UDR measures how quickly uncertainty decreases across conversation turns.
    Higher values indicate faster uncertainty reduction.
    
    Formula: UDR = -slope(uncertainty_over_time)
    
    Args:
        uncertainty_history: List of uncertainty values across turns
        window_size: Number of recent turns to consider
    
    Returns:
        UDR value (positive = decreasing uncertainty)
    """
    if len(uncertainty_history) < 2:
        return 0.0
    
    # Use only recent window
    recent_history = uncertainty_history[-window_size:] if len(uncertainty_history) > window_size else uncertainty_history
    
    if len(recent_history) < 2:
        return 0.0
    
    # Compute linear regression slope
    x = np.arange(len(recent_history))
    y = np.array(recent_history)
    
    # Handle edge cases
    if np.std(y) < 1e-6:
        return 0.0
    
    slope, _ = np.polyfit(x, y, 1)
    
    # Negative slope means decreasing uncertainty (positive UDR)
    udr = -slope
    
    return float(udr)


def compute_epistemic_convergence_speed(
    epistemic_history: List[float],
    convergence_threshold: float = 0.1
) -> float:
    """
    Compute Epistemic Convergence Speed (ECS).
    
    ECS measures how quickly epistemic uncertainty converges to a stable low value.
    Higher values indicate faster convergence.
    
    Args:
        epistemic_history: List of epistemic uncertainty values
        convergence_threshold: Threshold below which uncertainty is considered converged
    
    Returns:
        ECS value (number of turns to converge, or inf if not converged)
    """
    if len(epistemic_history) < 2:
        return 0.0
    
    # Find first turn where uncertainty drops below threshold
    for turn_idx, uncertainty in enumerate(epistemic_history):
        if uncertainty < convergence_threshold:
            # ECS = 1 / turns_to_converge (higher is faster)
            ecs = 1.0 / (turn_idx + 1)
            return ecs
    
    # Not converged
    return 0.0


def compute_routing_adaptation_score(
    routing_decisions: List[int],
    uncertainty_history: List[float],
    adaptation_threshold: float = 0.2
) -> float:
    """
    Compute Routing Adaptation Score (RAS).
    
    RAS measures how well routing decisions adapt to uncertainty changes.
    Higher values indicate better adaptation.
    
    Args:
        routing_decisions: List of routing decisions across turns
        uncertainty_history: List of total uncertainty values
        adaptation_threshold: Threshold for significant uncertainty change
    
    Returns:
        RAS value (0-1, higher is better)
    """
    if len(routing_decisions) < 2 or len(uncertainty_history) < 2:
        return 0.0
    
    adaptations = 0
    total_opportunities = 0
    
    for i in range(1, min(len(routing_decisions), len(uncertainty_history))):
        uncertainty_change = abs(uncertainty_history[i] - uncertainty_history[i-1])
        
        # If uncertainty changed significantly
        if uncertainty_change > adaptation_threshold:
            total_opportunities += 1
            
            # Did routing decision also change?
            if routing_decisions[i] != routing_decisions[i-1]:
                adaptations += 1
    
    if total_opportunities == 0:
        return 0.0
    
    ras = adaptations / total_opportunities
    return ras


def compute_temporal_metrics_batch(
    epistemic_histories: List[List[float]],
    aleatoric_histories: List[List[float]],
    routing_histories: List[List[int]],
    udr_window: int = 3,
    ecs_threshold: float = 0.1,
    ras_threshold: float = 0.2
) -> TemporalMetrics:
    """
    Compute temporal metrics for a batch of conversations.
    
    Args:
        epistemic_histories: List of epistemic uncertainty histories
        aleatoric_histories: List of aleatoric uncertainty histories
        routing_histories: List of routing decision histories
        udr_window: Window size for UDR computation
        ecs_threshold: Convergence threshold for ECS
        ras_threshold: Adaptation threshold for RAS
    
    Returns:
        TemporalMetrics object with aggregated metrics
    """
    udr_values = []
    ecs_values = []
    ras_values = []
    epistemic_stds = []
    aleatoric_stds = []
    total_uncertainties = []
    
    for ep_hist, al_hist, route_hist in zip(epistemic_histories, aleatoric_histories, routing_histories):
        # Compute total uncertainty
        total_hist = [np.sqrt(e**2 + a**2) for e, a in zip(ep_hist, al_hist)]
        
        # UDR
        udr = compute_uncertainty_decay_rate(total_hist, window_size=udr_window)
        udr_values.append(udr)
        
        # ECS
        ecs = compute_epistemic_convergence_speed(ep_hist, convergence_threshold=ecs_threshold)
        ecs_values.append(ecs)
        
        # RAS
        ras = compute_routing_adaptation_score(route_hist, total_hist, adaptation_threshold=ras_threshold)
        ras_values.append(ras)
        
        # Standard deviations
        epistemic_stds.append(np.std(ep_hist) if len(ep_hist) > 1 else 0.0)
        aleatoric_stds.append(np.std(al_hist) if len(al_hist) > 1 else 0.0)
        total_uncertainties.append(np.mean(total_hist))
    
    return TemporalMetrics(
        uncertainty_decay_rate=np.mean(udr_values),
        epistemic_convergence_speed=np.mean(ecs_values),
        routing_adaptation_score=np.mean(ras_values),
        epistemic_std=np.mean(epistemic_stds),
        aleatoric_std=np.mean(aleatoric_stds),
        total_uncertainty_mean=np.mean(total_uncertainties)
    )


def compute_calibration_metrics(
    uncertainties: np.ndarray,
    errors: np.ndarray,
    num_bins: int = 10
) -> Dict[str, float]:
    """
    Compute uncertainty calibration metrics.
    
    Calibration measures whether predicted uncertainties match actual errors.
    
    Args:
        uncertainties: Predicted uncertainties [num_samples]
        errors: Actual prediction errors [num_samples]
        num_bins: Number of bins for calibration curve
    
    Returns:
        Dictionary with calibration metrics
    """
    # Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (uncertainties >= bin_lower) & (uncertainties < bin_upper)
        
        if np.sum(in_bin) > 0:
            # Average uncertainty in bin
            avg_uncertainty = np.mean(uncertainties[in_bin])
            
            # Average error in bin
            avg_error = np.mean(errors[in_bin])
            
            # Contribution to ECE
            bin_weight = np.sum(in_bin) / len(uncertainties)
            ece += bin_weight * abs(avg_uncertainty - avg_error)
    
    # Correlation between uncertainty and error
    correlation = np.corrcoef(uncertainties, errors)[0, 1]
    
    return {
        'expected_calibration_error': ece,
        'uncertainty_error_correlation': correlation if not np.isnan(correlation) else 0.0
    }


def compare_models_significance(
    model1_results: np.ndarray,
    model2_results: np.ndarray,
    test_type: str = 'paired_t_test',
    alpha: float = 0.05
) -> Dict[str, any]:
    """
    Test statistical significance between two models.
    
    Args:
        model1_results: Results from model 1 [num_samples]
        model2_results: Results from model 2 [num_samples]
        test_type: Type of statistical test ('paired_t_test' or 'wilcoxon')
        alpha: Significance level
    
    Returns:
        Dictionary with test results
    """
    if test_type == 'paired_t_test':
        statistic, p_value = stats.ttest_rel(model1_results, model2_results)
        test_name = "Paired t-test"
    elif test_type == 'wilcoxon':
        statistic, p_value = stats.wilcoxon(model1_results, model2_results)
        test_name = "Wilcoxon signed-rank test"
    else:
        raise ValueError(f"Unknown test type: {test_type}")
    
    is_significant = p_value < alpha
    
    mean_diff = np.mean(model1_results) - np.mean(model2_results)
    
    return {
        'test_name': test_name,
        'statistic': float(statistic),
        'p_value': float(p_value),
        'is_significant': is_significant,
        'alpha': alpha,
        'mean_difference': float(mean_diff),
        'model1_mean': float(np.mean(model1_results)),
        'model2_mean': float(np.mean(model2_results)),
        'model1_std': float(np.std(model1_results)),
        'model2_std': float(np.std(model2_results))
    }


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic_fn: callable = np.mean,
    num_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.
    
    Args:
        data: Data array
        statistic_fn: Function to compute statistic
        num_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95%)
    
    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)
    """
    # Point estimate
    point_estimate = statistic_fn(data)
    
    # Bootstrap samples
    bootstrap_stats = []
    n = len(data)
    
    for _ in range(num_bootstrap):
        # Resample with replacement
        sample = np.random.choice(data, size=n, replace=True)
        stat = statistic_fn(sample)
        bootstrap_stats.append(stat)
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Compute confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_stats, lower_percentile)
    upper_bound = np.percentile(bootstrap_stats, upper_percentile)
    
    return point_estimate, lower_bound, upper_bound


def compute_conversation_level_metrics(
    conversation_predictions: List[List[int]],
    conversation_targets: List[List[int]],
    conversation_uncertainties: List[Dict[str, List[float]]]
) -> Dict[str, float]:
    """
    Compute metrics at conversation level (aggregated across turns).
    
    Args:
        conversation_predictions: List of prediction sequences
        conversation_targets: List of target sequences
        conversation_uncertainties: List of uncertainty dictionaries
    
    Returns:
        Dictionary of conversation-level metrics
    """
    metrics = {}
    
    # Flatten for turn-level metrics
    all_preds = [p for conv in conversation_predictions for p in conv]
    all_targets = [t for conv in conversation_targets for t in conv]
    
    # Standard metrics
    standard_metrics = compute_routing_metrics(
        np.array(all_preds),
        np.array(all_targets)
    )
    metrics.update(standard_metrics)
    
    # Temporal metrics
    epistemic_histories = [u['epistemic'] for u in conversation_uncertainties]
    aleatoric_histories = [u['aleatoric'] for u in conversation_uncertainties]
    
    temporal_metrics = compute_temporal_metrics_batch(
        epistemic_histories=epistemic_histories,
        aleatoric_histories=aleatoric_histories,
        routing_histories=conversation_predictions
    )
    
    metrics.update({
        'udr': temporal_metrics.uncertainty_decay_rate,
        'ecs': temporal_metrics.epistemic_convergence_speed,
        'ras': temporal_metrics.routing_adaptation_score,
        'epistemic_std': temporal_metrics.epistemic_std,
        'aleatoric_std': temporal_metrics.aleatoric_std,
        'total_uncertainty_mean': temporal_metrics.total_uncertainty_mean
    })
    
    # Conversation-level statistics
    conv_lengths = [len(conv) for conv in conversation_predictions]
    metrics['avg_conversation_length'] = np.mean(conv_lengths)
    metrics['num_conversations'] = len(conversation_predictions)
    
    return metrics


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate synthetic data
    predictions = np.random.randint(0, 4, size=100)
    targets = np.random.randint(0, 4, size=100)
    
    # Standard metrics
    metrics = compute_routing_metrics(predictions, targets)
    print("Standard Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Temporal metrics
    epistemic_history = [0.8, 0.6, 0.4, 0.3, 0.2]
    aleatoric_history = [0.5, 0.5, 0.4, 0.4, 0.3]
    routing_history = [1, 1, 0, 0, 3]
    
    udr = compute_uncertainty_decay_rate([np.sqrt(e**2 + a**2) 
                                           for e, a in zip(epistemic_history, aleatoric_history)])
    ecs = compute_epistemic_convergence_speed(epistemic_history)
    ras = compute_routing_adaptation_score(routing_history, 
                                           [np.sqrt(e**2 + a**2) 
                                            for e, a in zip(epistemic_history, aleatoric_history)])
    
    print(f"\nTemporal Metrics:")
    print(f"  UDR: {udr:.4f}")
    print(f"  ECS: {ecs:.4f}")
    print(f"  RAS: {ras:.4f}")
    
    # Statistical significance
    model1_scores = np.random.randn(100) + 0.75
    model2_scores = np.random.randn(100) + 0.70
    
    sig_test = compare_models_significance(model1_scores, model2_scores)
    print(f"\nStatistical Significance Test:")
    print(f"  Test: {sig_test['test_name']}")
    print(f"  p-value: {sig_test['p_value']:.4f}")
    print(f"  Significant: {sig_test['is_significant']}")
    print(f"  Mean difference: {sig_test['mean_difference']:.4f}")
    
    # Bootstrap CI
    point_est, lower, upper = bootstrap_confidence_interval(model1_scores)
    print(f"\nBootstrap 95% CI for Model 1:")
    print(f"  Point estimate: {point_est:.4f}")
    print(f"  CI: [{lower:.4f}, {upper:.4f}]")

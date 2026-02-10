"""
Comprehensive evaluator for Temporal Uncertainty Router.
Handles evaluation on test sets and comparison with baselines.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import logging
from pathlib import Path
import json

from ..models.temporal_router import TemporalUncertaintyRouter, ConversationState
from ..models.baselines import create_baseline_models
from .metrics import (
    compute_routing_metrics,
    compute_temporal_metrics_batch,
    compute_calibration_metrics,
    compare_models_significance,
    bootstrap_confidence_interval,
    compute_conversation_level_metrics
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalEvaluator:
    """
    Comprehensive evaluator for temporal uncertainty routing.
    
    Args:
        model: Trained TemporalUncertaintyRouter
        test_loader: Test data loader
        device: Device for evaluation
        output_dir: Directory to save results
    """
    
    def __init__(
        self,
        model: TemporalUncertaintyRouter,
        test_loader,
        device: str = 'cuda',
        output_dir: str = './results'
    ):
        self.model = model.to(device)
        self.model.eval()
        self.test_loader = test_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @torch.no_grad()
    def evaluate(self, compute_ci: bool = True) -> Dict:
        """
        Run comprehensive evaluation.
        
        Args:
            compute_ci: Whether to compute bootstrap confidence intervals
        
        Returns:
            Dictionary of evaluation results
        """
        logger.info("Running evaluation...")
        
        # Collect predictions and metrics
        all_predictions = []
        all_targets = []
        all_uncertainties = {'epistemic': [], 'aleatoric': []}
        all_temporal_metrics = {
            'udr': [], 'ecs': [], 
            'epistemic_trend': [], 'aleatoric_trend': []
        }
        
        # Track conversation-level data
        conversations = {}
        
        for batch in tqdm(self.test_loader, desc="Evaluating"):
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
                output['epistemic_uncertainty'].cpu().numpy().flatten()
            )
            all_uncertainties['aleatoric'].extend(
                output['aleatoric_uncertainty'].cpu().numpy().flatten()
            )
            
            # Collect temporal metrics
            for key in all_temporal_metrics.keys():
                all_temporal_metrics[key].extend(
                    output['temporal_metrics'][key].cpu().numpy().flatten()
                )
            
            # Group by conversation for conversation-level metrics
            for i, conv_id in enumerate(batch['conversation_id']):
                if conv_id not in conversations:
                    conversations[conv_id] = {
                        'predictions': [],
                        'targets': [],
                        'uncertainties': {'epistemic': [], 'aleatoric': []}
                    }
                
                conversations[conv_id]['predictions'].append(predictions[i].item())
                conversations[conv_id]['targets'].append(batch['routing_label'][i].item())
                conversations[conv_id]['uncertainties']['epistemic'].append(
                    output['epistemic_uncertainty'][i].item()
                )
                conversations[conv_id]['uncertainties']['aleatoric'].append(
                    output['aleatoric_uncertainty'][i].item()
                )
        
        # Convert to numpy
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Compute standard metrics
        logger.info("Computing standard metrics...")
        standard_metrics = compute_routing_metrics(all_predictions, all_targets)
        
        # Compute temporal metrics
        logger.info("Computing temporal metrics...")
        conv_predictions = [c['predictions'] for c in conversations.values()]
        conv_uncertainties = [c['uncertainties'] for c in conversations.values()]
        
        temporal_metrics = compute_temporal_metrics_batch(
            epistemic_histories=[u['epistemic'] for u in conv_uncertainties],
            aleatoric_histories=[u['aleatoric'] for u in conv_uncertainties],
            routing_histories=conv_predictions
        )
        
        # Compute calibration metrics
        logger.info("Computing calibration metrics...")
        total_uncertainties = np.sqrt(
            np.array(all_uncertainties['epistemic'])**2 + 
            np.array(all_uncertainties['aleatoric'])**2
        )
        prediction_errors = (all_predictions != all_targets).astype(float)
        
        calibration_metrics = compute_calibration_metrics(
            total_uncertainties,
            prediction_errors
        )
        
        # Confidence intervals (if requested)
        ci_metrics = {}
        if compute_ci:
            logger.info("Computing bootstrap confidence intervals...")
            
            # F1 score CI
            f1_point, f1_lower, f1_upper = bootstrap_confidence_interval(
                all_predictions == all_targets,
                statistic_fn=lambda x: np.mean(x),
                num_bootstrap=1000
            )
            ci_metrics['f1_ci'] = {
                'point': f1_point,
                'lower': f1_lower,
                'upper': f1_upper
            }
        
        # Combine all results
        results = {
            'standard_metrics': standard_metrics,
            'temporal_metrics': {
                'udr': temporal_metrics.uncertainty_decay_rate,
                'ecs': temporal_metrics.epistemic_convergence_speed,
                'ras': temporal_metrics.routing_adaptation_score,
                'epistemic_std': temporal_metrics.epistemic_std,
                'aleatoric_std': temporal_metrics.aleatoric_std,
                'total_uncertainty_mean': temporal_metrics.total_uncertainty_mean
            },
            'calibration_metrics': calibration_metrics,
            'confidence_intervals': ci_metrics,
            'summary': {
                'num_samples': len(all_predictions),
                'num_conversations': len(conversations),
                'avg_conversation_length': np.mean([len(c['predictions']) 
                                                     for c in conversations.values()])
            }
        }
        
        # Save results
        self._save_results(results)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def compare_with_baselines(
        self,
        baseline_models: Dict,
        compute_significance: bool = True
    ) -> Dict:
        """
        Compare temporal router with baseline models.
        
        Args:
            baseline_models: Dictionary of baseline models
            compute_significance: Whether to compute statistical significance
        
        Returns:
            Comparison results
        """
        logger.info("Comparing with baselines...")
        
        # Get temporal router results
        temporal_results = self.evaluate(compute_ci=False)
        temporal_scores = np.array([
            1.0 if p == t else 0.0 
            for p, t in zip(temporal_results['predictions'], 
                           temporal_results['targets'])
        ])
        
        comparison_results = {
            'temporal_router': temporal_results['standard_metrics']
        }
        
        # Evaluate each baseline
        for name, baseline in baseline_models.items():
            logger.info(f"Evaluating {name}...")
            
            if name == 'oracle':
                # Oracle uses ground truth
                baseline_predictions = temporal_results['targets']
            elif name in ['random', 'majority', 'heuristic']:
                # Non-neural baselines
                # Simplified - you'd need to pass appropriate inputs
                baseline_predictions = np.array([
                    baseline(turn_id=i % 5) if name == 'heuristic'
                    else baseline() for i in range(len(temporal_results['targets']))
                ])
            else:
                # Neural baselines - requires proper evaluation loop
                # Placeholder for now
                baseline_predictions = temporal_results['predictions']
            
            # Compute metrics
            baseline_metrics = compute_routing_metrics(
                baseline_predictions,
                temporal_results['targets']
            )
            
            comparison_results[name] = baseline_metrics
            
            # Statistical significance test
            if compute_significance and name != 'oracle':
                baseline_scores = (baseline_predictions == temporal_results['targets']).astype(float)
                
                sig_test = compare_models_significance(
                    temporal_scores,
                    baseline_scores,
                    test_type='paired_t_test'
                )
                
                comparison_results[f'{name}_significance'] = sig_test
        
        # Save comparison
        self._save_comparison(comparison_results)
        
        return comparison_results
    
    def analyze_uncertainty_evolution(self) -> Dict:
        """
        Analyze how uncertainty evolves across conversation turns.
        
        Returns:
            Analysis results
        """
        logger.info("Analyzing uncertainty evolution...")
        
        turn_uncertainties = {i: {'epistemic': [], 'aleatoric': []} 
                             for i in range(15)}
        turn_routing = {i: [] for i in range(15)}
        
        for batch in tqdm(self.test_loader, desc="Analyzing"):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            output = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            # Group by turn ID
            for i, turn_id in enumerate(batch['turn_id'].cpu().numpy()):
                if turn_id < 15:
                    turn_uncertainties[turn_id]['epistemic'].append(
                        output['epistemic_uncertainty'][i].item()
                    )
                    turn_uncertainties[turn_id]['aleatoric'].append(
                        output['aleatoric_uncertainty'][i].item()
                    )
                    turn_routing[turn_id].append(
                        torch.argmax(output['routing_logits'][i]).item()
                    )
        
        # Compute statistics per turn
        analysis = {}
        for turn_id in range(15):
            if turn_uncertainties[turn_id]['epistemic']:
                analysis[f'turn_{turn_id}'] = {
                    'epistemic_mean': np.mean(turn_uncertainties[turn_id]['epistemic']),
                    'epistemic_std': np.std(turn_uncertainties[turn_id]['epistemic']),
                    'aleatoric_mean': np.mean(turn_uncertainties[turn_id]['aleatoric']),
                    'aleatoric_std': np.std(turn_uncertainties[turn_id]['aleatoric']),
                    'routing_distribution': {
                        str(i): (np.array(turn_routing[turn_id]) == i).sum()
                        for i in range(4)
                    },
                    'num_samples': len(turn_uncertainties[turn_id]['epistemic'])
                }
        
        # Save analysis
        with open(self.output_dir / 'uncertainty_evolution.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Saved uncertainty evolution analysis to {self.output_dir / 'uncertainty_evolution.json'}")
        
        return analysis
    
    def _save_results(self, results: Dict):
        """Save evaluation results to file."""
        output_file = self.output_dir / 'evaluation_results.json'
        
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(results)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved results to {output_file}")
    
    def _save_comparison(self, comparison: Dict):
        """Save baseline comparison to file."""
        output_file = self.output_dir / 'baseline_comparison.json'
        
        with open(output_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        logger.info(f"Saved comparison to {output_file}")
    
    def _print_summary(self, results: Dict):
        """Print evaluation summary."""
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 80)
        
        logger.info("\nStandard Metrics:")
        for key, value in results['standard_metrics'].items():
            if not key.startswith('f1_class'):
                logger.info(f"  {key}: {value:.4f}")
        
        logger.info("\nTemporal Metrics:")
        for key, value in results['temporal_metrics'].items():
            logger.info(f"  {key}: {value:.4f}")
        
        logger.info("\nCalibration Metrics:")
        for key, value in results['calibration_metrics'].items():
            logger.info(f"  {key}: {value:.4f}")
        
        logger.info("\nSummary:")
        for key, value in results['summary'].items():
            logger.info(f"  {key}: {value}")
        
        logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    # Example usage
    from transformers import AutoTokenizer
    from ..models.temporal_router import TemporalUncertaintyRouter
    from ..data.dataloader import ConversationDataLoader, create_dataloaders
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model = TemporalUncertaintyRouter(
        encoder_name='bert-base-uncased',
        embedding_dim=768,
        hidden_dim=256
    )
    
    # Load test data
    data_loader = ConversationDataLoader(dataset_name='coqa')
    _, val_convs = data_loader.load_and_preprocess()
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    _, test_loader = create_dataloaders(
        val_convs[:50], val_convs[:10],
        tokenizer, batch_size=8
    )
    
    # Create evaluator
    evaluator = TemporalEvaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir='./results/test'
    )
    
    # Run evaluation
    results = evaluator.evaluate()
    
    # Analyze uncertainty evolution
    evolution = evaluator.analyze_uncertainty_evolution()

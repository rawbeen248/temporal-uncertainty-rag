"""
Baseline models for comparison with Temporal Uncertainty Router.

Includes:
1. Random Router
2. Static Router (no temporal features)
3. Uncertainty-Only Router
4. Oracle Router (upper bound)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
from transformers import AutoModel


class RandomRouter:
    """
    Baseline: Random routing.
    Routes queries randomly to any source.
    """
    
    def __init__(self, num_sources: int = 4, seed: int = 42):
        self.num_sources = num_sources
        self.rng = np.random.RandomState(seed)
    
    def __call__(self, *args, **kwargs) -> int:
        """Make random routing decision."""
        return self.rng.randint(0, self.num_sources)
    
    def predict_batch(self, batch_size: int) -> np.ndarray:
        """Predict for a batch."""
        return self.rng.randint(0, self.num_sources, size=batch_size)


class StaticRouter(nn.Module):
    """
    Baseline: Static single-turn router.
    Routes based only on current query, no conversation history.
    """
    
    def __init__(
        self,
        encoder_name: str = 'bert-base-uncased',
        embedding_dim: int = 768,
        hidden_dim: int = 256,
        num_sources: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(encoder_name)
        
        # Simple routing network
        self.router = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_sources)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            routing_logits: [batch_size, num_sources]
        """
        # Encode query
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        query_embedding = encoder_output.last_hidden_state[:, 0, :]
        
        # Route
        routing_logits = self.router(query_embedding)
        
        return routing_logits


class UncertaintyOnlyRouter(nn.Module):
    """
    Baseline: Routes based only on current uncertainty.
    No temporal tracking, just instant uncertainty.
    """
    
    def __init__(
        self,
        encoder_name: str = 'bert-base-uncased',
        embedding_dim: int = 768,
        hidden_dim: int = 256,
        num_sources: int = 4,
        dropout: float = 0.1,
        num_mc_samples: int = 10
    ):
        super().__init__()
        
        self.num_mc_samples = num_mc_samples
        
        self.encoder = AutoModel.from_pretrained(encoder_name)
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # epistemic + aleatoric
        )
        
        # Router (based on query + uncertainty)
        self.router = nn.Sequential(
            nn.Linear(embedding_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_sources)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        # Encode
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        query_embedding = encoder_output.last_hidden_state[:, 0, :]
        
        # Estimate uncertainty
        uncertainties = self.uncertainty_head(query_embedding)
        epistemic = uncertainties[:, 0:1]
        aleatoric = uncertainties[:, 1:2]
        
        # Route based on query + uncertainty
        router_input = torch.cat([query_embedding, epistemic, aleatoric], dim=1)
        routing_logits = self.router(router_input)
        
        return {
            'routing_logits': routing_logits,
            'epistemic': epistemic,
            'aleatoric': aleatoric
        }


class OracleRouter:
    """
    Oracle baseline: Perfect routing using ground truth.
    
    This represents an upper bound on performance.
    In practice, uses ground truth labels directly.
    """
    
    def __init__(self):
        pass
    
    def __call__(self, ground_truth: int) -> int:
        """Return ground truth routing decision."""
        return ground_truth
    
    def predict_batch(self, ground_truths: np.ndarray) -> np.ndarray:
        """Predict for a batch using ground truth."""
        return ground_truths


class MajorityClassRouter:
    """
    Baseline: Always routes to most common source.
    """
    
    def __init__(self, majority_class: int = 0):
        self.majority_class = majority_class
    
    def __call__(self, *args, **kwargs) -> int:
        """Always return majority class."""
        return self.majority_class
    
    def predict_batch(self, batch_size: int) -> np.ndarray:
        """Predict for a batch."""
        return np.full(batch_size, self.majority_class)
    
    def fit(self, targets: np.ndarray):
        """Fit by finding most common class in training data."""
        unique, counts = np.unique(targets, return_counts=True)
        self.majority_class = unique[np.argmax(counts)]
        return self


class HeuristicRouter:
    """
    Baseline: Rule-based heuristic routing.
    
    Rules:
    - Turn 0: External search
    - Unanswerable queries: Clarification
    - Turn 3+: Multi-source fusion
    - Otherwise: Internal KB
    """
    
    def __init__(self):
        pass
    
    def __call__(
        self,
        turn_id: int,
        is_answerable: bool = True
    ) -> int:
        """
        Route based on simple heuristics.
        
        Returns:
            0: Internal KB
            1: External Search
            2: Clarification
            3: Multi-source Fusion
        """
        if not is_answerable:
            return 2  # Clarification
        elif turn_id == 0:
            return 1  # External search
        elif turn_id >= 3:
            return 3  # Multi-source fusion
        else:
            return 0  # Internal KB


class EnsembleRouter(nn.Module):
    """
    Baseline: Ensemble of multiple routers.
    Combines predictions via voting or averaging.
    """
    
    def __init__(
        self,
        routers: list,
        combination: str = 'vote'  # 'vote' or 'average'
    ):
        super().__init__()
        self.routers = nn.ModuleList(routers)
        self.combination = combination
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Combine predictions from all routers."""
        logits_list = []
        
        for router in self.routers:
            if hasattr(router, 'forward'):
                output = router(*args, **kwargs)
                if isinstance(output, dict):
                    logits_list.append(output['routing_logits'])
                else:
                    logits_list.append(output)
        
        if self.combination == 'average':
            # Average logits
            combined_logits = torch.stack(logits_list).mean(dim=0)
        else:  # vote
            # Majority voting
            predictions = torch.stack([torch.argmax(l, dim=-1) for l in logits_list])
            # Convert votes to logits (one-hot encoding of majority)
            combined_logits = torch.zeros_like(logits_list[0])
            for i in range(combined_logits.size(0)):
                majority_vote = torch.mode(predictions[:, i]).values
                combined_logits[i, majority_vote] = 1.0
        
        return combined_logits


def create_baseline_models(
    num_sources: int = 4,
    encoder_name: str = 'bert-base-uncased',
    device: str = 'cuda'
) -> Dict[str, any]:
    """
    Create all baseline models for comparison.
    
    Args:
        num_sources: Number of routing sources
        encoder_name: Pre-trained encoder name
        device: Device to put models on
    
    Returns:
        Dictionary of baseline models
    """
    baselines = {
        'random': RandomRouter(num_sources=num_sources),
        'majority': MajorityClassRouter(),
        'heuristic': HeuristicRouter(),
        'oracle': OracleRouter()
    }
    
    # Neural baselines
    static = StaticRouter(
        encoder_name=encoder_name,
        num_sources=num_sources
    ).to(device)
    
    uncertainty_only = UncertaintyOnlyRouter(
        encoder_name=encoder_name,
        num_sources=num_sources
    ).to(device)
    
    baselines['static'] = static
    baselines['uncertainty_only'] = uncertainty_only
    
    return baselines


if __name__ == "__main__":
    # Example usage
    from transformers import AutoTokenizer
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create baselines
    baselines = create_baseline_models(device=device)
    
    print("Created baselines:")
    for name, model in baselines.items():
        if isinstance(model, nn.Module):
            num_params = sum(p.numel() for p in model.parameters())
            print(f"  {name}: {num_params:,} parameters")
        else:
            print(f"  {name}: Non-parametric")
    
    # Test random router
    random_router = baselines['random']
    predictions = random_router.predict_batch(10)
    print(f"\nRandom router predictions: {predictions}")
    
    # Test heuristic router
    heuristic_router = baselines['heuristic']
    print(f"\nHeuristic router:")
    print(f"  Turn 0: {heuristic_router(turn_id=0)}")
    print(f"  Turn 1: {heuristic_router(turn_id=1)}")
    print(f"  Turn 4: {heuristic_router(turn_id=4)}")
    print(f"  Unanswerable: {heuristic_router(turn_id=1, is_answerable=False)}")
    
    # Test static router
    static_router = baselines['static']
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    text = "What is the capital of France?"
    encoding = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    encoding = {k: v.to(device) for k, v in encoding.items()}
    
    with torch.no_grad():
        logits = static_router(**encoding)
        print(f"\nStatic router logits: {logits}")
        print(f"Predicted route: {torch.argmax(logits, dim=-1).item()}")

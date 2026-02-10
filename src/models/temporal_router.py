"""
Main Temporal Uncertainty Router model.
Routes queries based on temporal uncertainty evolution patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from transformers import AutoModel, AutoTokenizer

from .uncertainty_estimator import UncertaintyEstimator, TemporalUncertaintyTracker


class TemporalUncertaintyRouter(nn.Module):
    """
    Conversation-aware routing model that adapts based on uncertainty evolution.
    
    Architecture:
    1. BERT encoder for query understanding
    2. LSTM for conversation history encoding
    3. Uncertainty estimator (epistemic + aleatoric)
    4. Temporal tracker (UDR, ECS computation)
    5. Router (decision network)
    
    Args:
        encoder_name: Pre-trained encoder model name
        embedding_dim: Dimension of embeddings
        hidden_dim: Hidden dimension for LSTM and networks
        num_lstm_layers: Number of LSTM layers
        num_sources: Number of routing options (default: 4)
        dropout: Dropout rate
        num_mc_samples: MC dropout samples for uncertainty
    """
    
    def __init__(
        self,
        encoder_name: str = 'bert-base-uncased',
        embedding_dim: int = 768,
        hidden_dim: int = 256,
        num_lstm_layers: int = 2,
        num_sources: int = 4,
        dropout: float = 0.1,
        num_mc_samples: int = 10
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_sources = num_sources
        
        # Query encoder (BERT)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        
        # Freeze lower layers of BERT (optional, for faster training)
        for param in list(self.encoder.parameters())[:8]:
            param.requires_grad = False
        
        # Conversation history encoder (LSTM)
        self.history_encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Uncertainty estimator
        self.uncertainty_estimator = UncertaintyEstimator(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_mc_samples=num_mc_samples,
            dropout_rate=dropout
        )
        
        # Temporal uncertainty tracker
        self.temporal_tracker = TemporalUncertaintyTracker(window_size=5)
        
        # Router decision network
        router_input_dim = (
            embedding_dim +  # current query encoding
            hidden_dim * 2 +  # bidirectional LSTM output
            2 +  # epistemic + aleatoric uncertainty
            4   # temporal metrics (UDR, ECS, 2 trends)
        )
        
        self.router = nn.Sequential(
            nn.Linear(router_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_sources)
        )
        
        # Confidence head (for routing confidence estimation)
        self.confidence_head = nn.Sequential(
            nn.Linear(router_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        history_embeddings: Optional[torch.Tensor] = None,
        history_uncertainties: Optional[Dict[str, torch.Tensor]] = None,
        return_all: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for routing decision.
        
        Args:
            input_ids: Tokenized input [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            history_embeddings: Previous turn embeddings [batch_size, num_prev_turns, embedding_dim]
            history_uncertainties: Dict with 'epistemic' and 'aleatoric' history
            return_all: Whether to return all intermediate outputs
        
        Returns:
            Dictionary containing routing logits and uncertainties
        """
        batch_size = input_ids.size(0)
        
        # 1. Encode current query
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        query_embedding = encoder_output.last_hidden_state[:, 0, :]  # CLS token
        
        # 2. Encode conversation history
        if history_embeddings is not None and history_embeddings.size(1) > 0:
            # LSTM over history
            history_output, (h_n, c_n) = self.history_encoder(history_embeddings)
            # Use final hidden state (concatenated forward + backward)
            history_context = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [batch_size, hidden_dim*2]
        else:
            # No history (first turn)
            history_context = torch.zeros(batch_size, self.hidden_dim * 2, device=input_ids.device)
        
        # 3. Estimate current uncertainty
        current_epistemic, current_aleatoric, _ = self.uncertainty_estimator(query_embedding)
        
        # 4. Compute temporal uncertainty metrics
        if history_uncertainties is not None:
            temporal_metrics = self.temporal_tracker(
                current_epistemic=current_epistemic,
                current_aleatoric=current_aleatoric,
                history_epistemic=history_uncertainties.get('epistemic'),
                history_aleatoric=history_uncertainties.get('aleatoric')
            )
        else:
            # First turn - no temporal metrics
            temporal_metrics = {
                'udr': torch.zeros_like(current_epistemic),
                'ecs': torch.zeros_like(current_epistemic),
                'epistemic_trend': torch.zeros_like(current_epistemic),
                'aleatoric_trend': torch.zeros_like(current_aleatoric)
            }
        
        # 5. Concatenate all features for routing decision
        router_features = torch.cat([
            query_embedding,                          # [batch_size, embedding_dim]
            history_context,                          # [batch_size, hidden_dim*2]
            current_epistemic,                        # [batch_size, 1]
            current_aleatoric,                        # [batch_size, 1]
            temporal_metrics['udr'],                  # [batch_size, 1]
            temporal_metrics['ecs'],                  # [batch_size, 1]
            temporal_metrics['epistemic_trend'],      # [batch_size, 1]
            temporal_metrics['aleatoric_trend']       # [batch_size, 1]
        ], dim=1)
        
        # 6. Make routing decision
        routing_logits = self.router(router_features)
        routing_probs = F.softmax(routing_logits, dim=-1)
        
        # 7. Estimate routing confidence
        routing_confidence = self.confidence_head(router_features)
        
        # Prepare output
        output = {
            'routing_logits': routing_logits,
            'routing_probs': routing_probs,
            'routing_confidence': routing_confidence,
            'epistemic_uncertainty': current_epistemic,
            'aleatoric_uncertainty': current_aleatoric,
            'temporal_metrics': temporal_metrics
        }
        
        if return_all:
            output.update({
                'query_embedding': query_embedding,
                'history_context': history_context,
                'router_features': router_features
            })
        
        return output
    
    def route(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        history_embeddings: Optional[torch.Tensor] = None,
        history_uncertainties: Optional[Dict[str, torch.Tensor]] = None,
        threshold: float = 0.5
    ) -> Tuple[int, float]:
        """
        Make routing decision with confidence threshold.
        
        Args:
            input_ids: Tokenized input
            attention_mask: Attention mask
            history_embeddings: Previous turn embeddings
            history_uncertainties: Historical uncertainties
            threshold: Confidence threshold for routing
        
        Returns:
            Tuple of (route_id, confidence)
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                history_embeddings=history_embeddings,
                history_uncertainties=history_uncertainties
            )
            
            routing_probs = output['routing_probs']
            confidence = output['routing_confidence']
            
            # Get most confident route
            route_id = torch.argmax(routing_probs, dim=-1).item()
            route_confidence = routing_probs[0, route_id].item()
            
            # If confidence below threshold, route to clarification
            if route_confidence < threshold:
                route_id = 2  # Clarification
            
            return route_id, route_confidence


class StaticRouter(nn.Module):
    """
    Baseline static router (no temporal features).
    Routes based only on current query.
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
        
        # Query encoder
        self.encoder = AutoModel.from_pretrained(encoder_name)
        
        # Router
        self.router = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_sources)
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass for static routing."""
        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        query_embedding = encoder_output.last_hidden_state[:, 0, :]
        routing_logits = self.router(query_embedding)
        return routing_logits


class ConversationState:
    """
    Maintains conversation state for multi-turn routing.
    """
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.embeddings = []
        self.uncertainties = {'epistemic': [], 'aleatoric': []}
        self.routing_decisions = []
    
    def update(
        self,
        embedding: torch.Tensor,
        epistemic: torch.Tensor,
        aleatoric: torch.Tensor,
        routing_decision: int
    ):
        """Update conversation state with new turn."""
        self.embeddings.append(embedding)
        self.uncertainties['epistemic'].append(epistemic)
        self.uncertainties['aleatoric'].append(aleatoric)
        self.routing_decisions.append(routing_decision)
        
        # Maintain max history
        if len(self.embeddings) > self.max_history:
            self.embeddings.pop(0)
            self.uncertainties['epistemic'].pop(0)
            self.uncertainties['aleatoric'].pop(0)
            self.routing_decisions.pop(0)
    
    def get_history_embeddings(self) -> Optional[torch.Tensor]:
        """Get stacked history embeddings."""
        if not self.embeddings:
            return None
        return torch.stack(self.embeddings, dim=0).unsqueeze(0)  # [1, num_turns, embedding_dim]
    
    def get_history_uncertainties(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get stacked history uncertainties."""
        if not self.uncertainties['epistemic']:
            return None
        
        return {
            'epistemic': torch.stack(self.uncertainties['epistemic'], dim=0).unsqueeze(0),
            'aleatoric': torch.stack(self.uncertainties['aleatoric'], dim=0).unsqueeze(0)
        }
    
    def reset(self):
        """Reset conversation state."""
        self.embeddings = []
        self.uncertainties = {'epistemic': [], 'aleatoric': []}
        self.routing_decisions = []


if __name__ == "__main__":
    # Example usage
    from transformers import AutoTokenizer
    
    # Initialize model
    model = TemporalUncertaintyRouter(
        encoder_name='bert-base-uncased',
        embedding_dim=768,
        hidden_dim=256,
        num_sources=4
    )
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Example query
    query = "What is the capital of France?"
    context = "France is a country in Europe."
    input_text = f"Context: {context} Question: {query}"
    
    # Tokenize
    encoding = tokenizer(
        input_text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Forward pass (first turn, no history)
    output = model(
        input_ids=encoding['input_ids'],
        attention_mask=encoding['attention_mask']
    )
    
    print("Output keys:", output.keys())
    print(f"Routing logits: {output['routing_logits']}")
    print(f"Routing probs: {output['routing_probs']}")
    print(f"Epistemic uncertainty: {output['epistemic_uncertainty'].item():.4f}")
    print(f"Aleatoric uncertainty: {output['aleatoric_uncertainty'].item():.4f}")
    
    # Simulate second turn with history
    conv_state = ConversationState()
    conv_state.update(
        embedding=output['query_embedding'].squeeze(0),
        epistemic=output['epistemic_uncertainty'].squeeze(0),
        aleatoric=output['aleatoric_uncertainty'].squeeze(0),
        routing_decision=torch.argmax(output['routing_logits']).item()
    )
    
    # Second query
    query2 = "What is its population?"
    input_text2 = f"Context: {context} Question: {query2}"
    encoding2 = tokenizer(input_text2, max_length=512, padding='max_length', 
                          truncation=True, return_tensors='pt')
    
    output2 = model(
        input_ids=encoding2['input_ids'],
        attention_mask=encoding2['attention_mask'],
        history_embeddings=conv_state.get_history_embeddings(),
        history_uncertainties=conv_state.get_history_uncertainties()
    )
    
    print(f"\n=== Turn 2 ===")
    print(f"Routing probs: {output2['routing_probs']}")
    print(f"UDR: {output2['temporal_metrics']['udr'].item():.4f}")
    print(f"ECS: {output2['temporal_metrics']['ecs'].item():.4f}")
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

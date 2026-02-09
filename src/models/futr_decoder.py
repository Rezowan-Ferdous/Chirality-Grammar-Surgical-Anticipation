"""
FUTR decoder for multi-task action anticipation.

Implements transformer-based decoder with 4 parallel prediction heads:
- Action head (categorical)
- Duration head (regression)
- Object head (multi-label binary)
- Goal head (optional)

Paper: "6-layer Transformer decoder with d=512, 8 attention heads"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            (B, T, D) with positional encoding added
        """
        return x + self.pe[:, :x.size(1)]


class FUTRDecoder(nn.Module):
    """
    Future Transformer (FUTR) decoder for action anticipation.
    
    Paper: "FUTR uses a 6-layer Transformer decoder with self-attention
    and cross-attention to V-JEPA embeddings"
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_actions: int = 466,
        num_objects: int = 16,
        num_goals: int = 8,
        use_goal_head: bool = False,
        max_horizon: int = 50
    ):
        """
        Args:
            input_dim: Input feature dimension from V-JEPA
            hidden_dim: Hidden dimension (paper uses 512)
            num_layers: Number of transformer layers (paper uses 6)
            num_heads: Number of attention heads (paper uses 8)
            dim_feedforward: FFN dimension
            dropout: Dropout rate
            num_actions: Number of action classes
            num_objects: Number of object states
            num_goals: Number of goal states
            use_goal_head: Whether to include goal prediction head
            max_horizon: Maximum prediction horizon
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.max_horizon = max_horizon
        self.use_goal_head = use_goal_head
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # Future query embeddings (learnable)
        self.future_queries = nn.Parameter(torch.randn(max_horizon, hidden_dim))
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Prediction heads
        self.action_head = nn.Linear(hidden_dim, num_actions)
        self.duration_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.object_head = nn.Linear(hidden_dim, num_objects)
        
        if use_goal_head:
            self.goal_head = nn.Linear(hidden_dim, num_goals)
    
    def forward(
        self,
        encoder_features: torch.Tensor,
        prediction_horizon: int = 1,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            encoder_features: (B, T_obs, D) V-JEPA features from observed frames
            prediction_horizon: Number of future steps to predict
            encoder_mask: (B, T_obs) mask for padding
            
        Returns:
            Dictionary with predictions:
                - action_logits: (B, prediction_horizon, num_actions)
                - duration_pred: (B, prediction_horizon)
                - object_logits: (B, prediction_horizon, num_objects)
                - goal_logits: (B, prediction_horizon, num_goals) [optional]
        """
        B, T_obs, _ = encoder_features.shape
        
        # Project encoder features
        memory = self.input_proj(encoder_features)  # (B, T_obs, hidden_dim)
        memory = self.pos_encoding(memory)
        
        # Get future queries
        queries = self.future_queries[:prediction_horizon].unsqueeze(0)  # (1, horizon, hidden_dim)
        queries = queries.expand(B, -1, -1)  # (B, horizon, hidden_dim)
        
        # Transformer decoding
        # Self-attention on future queries + cross-attention to observed features
        decoded = self.transformer_decoder(
            tgt=queries,
            memory=memory,
            memory_key_padding_mask=encoder_mask
        )  # (B, horizon, hidden_dim)
        
        # Multi-task prediction
        outputs = {}
        
        # Action prediction (categorical)
        outputs['action_logits'] = self.action_head(decoded)  # (B, horizon, num_actions)
        
        # Duration prediction (regression)
        outputs['duration_pred'] = self.duration_head(decoded).squeeze(-1)  # (B, horizon)
        
        # Object prediction (multi-label binary)
        outputs['object_logits'] = self.object_head(decoded)  # (B, horizon, num_objects)
        
        # Goal prediction (optional)
        if self.use_goal_head:
            outputs['goal_logits'] = self.goal_head(decoded)  # (B, horizon, num_goals)
        
        # Decoder features for grammar scoring
        outputs['decoder_features'] = decoded
        
        return outputs
    
    def predict_next(
        self,
        encoder_features: torch.Tensor,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict next action, duration, and objects.
        
        Args:
            encoder_features: (B, T_obs, D)
            temperature: Sampling temperature for actions
            
        Returns:
            - action_probs: (B, num_actions)
            - duration: (B,)
            - object_probs: (B, num_objects)
        """
        outputs = self.forward(encoder_features, prediction_horizon=1)
        
        # Action distribution with temperature
        action_logits = outputs['action_logits'][:, 0] / temperature  # (B, num_actions)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Duration
        duration = outputs['duration_pred'][:, 0]  # (B,)
        
        # Object probabilities
        object_logits = outputs['object_logits'][:, 0]  # (B, num_objects)
        object_probs = torch.sigmoid(object_logits)
        
        return action_probs, duration, object_probs


# Example usage
if __name__ == '__main__':
    # Create decoder
    decoder = FUTRDecoder(
        input_dim=512,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        num_actions=466,
        num_objects=16,
        use_goal_head=True
    )
    
    # Test forward pass
    B, T_obs, D = 4, 16, 512
    encoder_features = torch.randn(B, T_obs, D)
    
    # Predict future actions
    outputs = decoder(encoder_features, prediction_horizon=10)
    
    print("FUTR Decoder outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # Test next action prediction
    action_probs, duration, object_probs = decoder.predict_next(encoder_features)
    print(f"\nNext action prediction:")
    print(f"  Action probs: {action_probs.shape}")
    print(f"  Duration: {duration.shape}")
    print(f"  Object probs: {object_probs.shape}")

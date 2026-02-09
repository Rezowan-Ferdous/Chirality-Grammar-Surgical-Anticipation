"""
Complete PTG model integrating V-JEPA encoder, FUTR decoder, and grammar.

Paper: "PTG unifies a neural stage (V-JEPA + FUTR) with symbolic grammar
through closed-loop training and inference"
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import torch.nn.functional as F

from .vjepa_extractor import VJEPAExtractor
from .futr_decoder import FUTRDecoder


class PTGModel(nn.Module):
    """
    Complete Probabilistic Temporal Grammar model.
    
    Architecture:
        1. V-JEPA Encoder (frozen, 1024-d)
        2. Linear Projection (512-d)
        3. FUTR Decoder (6 layers, 4 heads)
        4. Grammar Scorer (optional, for Phase 2)
    """
    
    def __init__(
        self,
        vjepa_model_name: str = "facebook/vjepa2-vitl-fpc64-256",
        freeze_vjepa: bool = True,
        hidden_dim: int = 512,
        num_decoder_layers: int = 6,
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
            vjepa_model_name: HuggingFace V-JEPA model name
            freeze_vjepa: Freeze V-JEPA backbone
            hidden_dim: Hidden dimension (paper: 512)
            num_decoder_layers: FUTR layers (paper: 6)
            num_heads: Attention heads (paper: 8)
            dim_feedforward: FFN dimension
            dropout: Dropout rate
            num_actions: Number of action classes
            num_objects: Number of object states
            num_goals: Number of goal states
            use_goal_head: Include goal prediction
            max_horizon: Maximum prediction horizon
        """
        super().__init__()
        
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        
        # V-JEPA Encoder
        self.vjepa_encoder = VJEPAExtractor(
            model_name=vjepa_model_name,
            freeze_backbone=freeze_vjepa,
            output_dim=hidden_dim
        )
        
        # FUTR Decoder
        self.futr_decoder = FUTRDecoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_actions=num_actions,
            num_objects=num_objects,
            num_goals=num_goals,
            use_goal_head=use_goal_head,
            max_horizon=max_horizon
        )
        
        # Grammar scorer (set later for Phase 2)
        self.grammar_scorer = None
    
    def forward(
        self,
        video_frames: torch.Tensor,
        prediction_horizon: int = 1,
        encoder_mask: Optional[torch.Tensor] = None,
        history: Optional[Tuple[str, ...]] = None,
        use_grammar: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            video_frames: (B, T_obs, 3, H, W) observed video frames
            prediction_horizon: Number of future steps to predict
            encoder_mask: (B, T_obs) padding mask
            history: Previous action history for grammar scoring
            use_grammar: Whether to compute grammar distribution (Phase 2)
            
        Returns:
            Dictionary with predictions and optional grammar distribution
        """
        # Extract V-JEPA features
        encoder_features = self.vjepa_encoder(video_frames)  # (B, T_obs, hidden_dim)
        
        # FUTR decoding
        outputs = self.futr_decoder(
            encoder_features,
            prediction_horizon=prediction_horizon,
            encoder_mask=encoder_mask
        )
        
        # Grammar scoring (Phase 2)
        if use_grammar and self.grammar_scorer is not None and history is not None:
            # Get next action prediction
            with torch.no_grad():
                action_probs, duration, object_probs = self.futr_decoder.predict_next(encoder_features)
            
            # Compute grammar distribution
            grammar_dist = self.grammar_scorer.get_distribution_torch(
                history=history,
                predicted_duration=duration,
                predicted_objects=object_probs
            )
            
            outputs['grammar_probs'] = grammar_dist
        
        return outputs
    
    def set_grammar_scorer(self, grammar_scorer):
        """Set grammar scorer for Phase 2 training."""
        self.grammar_scorer = grammar_scorer
    
    def predict_sequence(
        self,
        video_frames: torch.Tensor,
        max_steps: int = 10,
        use_grammar: bool = False,
        temperature: float = 1.0
    ) -> Dict[str, list]:
        """
        Autoregressively predict future action sequence.
        
        Args:
            video_frames: (1, T_obs, 3, H, W) single video
            max_steps: Maximum prediction steps
            use_grammar: Use grammar for scoring
            temperature: Sampling temperature
            
        Returns:
            Dictionary with predicted sequences
        """
        self.eval()
        
        predictions = {
            'actions': [],
            'durations': [],
            'objects': []
        }
        
        with torch.no_grad():
            # Extract features once
            encoder_features = self.vjepa_encoder(video_frames)  # (1, T_obs, D)
            
            history = tuple()  # Action history
            
            for step in range(max_steps):
                # Predict next action
                action_probs, duration, object_probs = self.futr_decoder.predict_next(
                    encoder_features,
                    temperature=temperature
                )
                
                # Sample action
                action_idx = torch.multinomial(action_probs[0], num_samples=1).item()
                
                predictions['actions'].append(action_idx)
                predictions['durations'].append(duration[0].item())
                predictions['objects'].append(object_probs[0].cpu().numpy())
                
                # Update history (placeholder - need action vocabulary)
                # history = history + (f'action_{action_idx}',)
        
        return predictions


# Example usage
if __name__ == '__main__':
    # Create PTG model
    model = PTGModel(
        vjepa_model_name="facebook/vjepa2-vitl-fpc64-256",
        freeze_vjepa=True,
        hidden_dim=512,
        num_decoder_layers=6,
        num_heads=8,
        num_actions=466,
        num_objects=16,
        use_goal_head=True
    )
    
    print(f"PTG Model created")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    B, T_obs, C, H, W = 2, 16, 3, 224, 224
    video_frames = torch.randn(B, T_obs, C, H, W)
    
    outputs = model(video_frames, prediction_horizon=10)
    
    print("\nModel outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # Test sequence prediction
    single_video = video_frames[:1]
    sequence = model.predict_sequence(single_video, max_steps=5)
    
    print("\nPredicted sequence:")
    print(f"  Actions: {sequence['actions']}")
    print(f"  Durations: {[f'{d:.1f}' for d in sequence['durations']]}")

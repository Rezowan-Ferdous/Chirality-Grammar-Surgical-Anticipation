"""
Multi-task losses for PTG training.

Implements:
1. Phase 1: Supervised pre-training loss
2. Phase 2: Grammar-regularized loss with KL divergence

Paper: "L_total = L_base + λ_gram L_grammar"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ActionLoss(nn.Module):
    """Cross-entropy loss for action classification."""
    
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, num_actions) or (B, horizon, num_actions)
            targets: (B,) or (B, horizon)
        """
        if logits.dim() == 3:
            # Multiple horizons
            B, H, C = logits.shape
            logits = logits.reshape(B * H, C)
            targets = targets.reshape(B * H)
        
        return self.criterion(logits, targets)


class DurationLoss(nn.Module):
    """MSE loss for duration regression."""
    
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            predictions: (B,) or (B, horizon)
            targets: (B,) or (B, horizon)
            mask: (B,) or (B, horizon) optional mask
        """
        if mask is not None:
            loss = F.mse_loss(predictions, targets, reduction='none')
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = self.criterion(predictions, targets)
        
        return loss


class ObjectLoss(nn.Module):
    """Binary cross-entropy loss for multi-label object prediction."""
    
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, num_objects) or (B, horizon, num_objects)
            targets: (B, num_objects) or (B, horizon, num_objects)
        """
        return self.criterion(logits, targets)


class GoalLoss(nn.Module):
    """Cross-entropy loss for goal classification (optional)."""
    
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, num_goals) or (B, horizon, num_goals)
            targets: (B,) or (B, horizon)
        """
        if logits.dim() == 3:
            B, H, C = logits.shape
            logits = logits.reshape(B * H, C)
            targets = targets.reshape(B * H)
        
        return self.criterion(logits, targets)


class GrammarKLLoss(nn.Module):
    """
    KL divergence loss for grammar regularization.
    
    Paper: "L_grammar = KL(S_N || P_G)"
    where S_N is neural distribution and P_G is grammar distribution
    """
    
    def __init__(self, conflict_threshold: float = 1e-5):
        """
        Args:
            conflict_threshold: Threshold for conflict masking
        """
        super().__init__()
        self.conflict_threshold = conflict_threshold
    
    def forward(
        self,
        neural_logits: torch.Tensor,
        grammar_probs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute KL(neural || grammar).
        
        Args:
            neural_logits: (B, num_actions) logits from neural model
            grammar_probs: (B, num_actions) probabilities from grammar
            mask: (B,) optional mask for conflict resolution
            
        Returns:
            KL divergence loss
        """
        # Neural distribution
        neural_probs = F.softmax(neural_logits, dim=-1)  # (B, num_actions)
        
        # KL divergence: KL(P || Q) = Σ P(x) log(P(x) / Q(x))
        kl_div = neural_probs * (torch.log(neural_probs + 1e-10) - torch.log(grammar_probs + 1e-10))
        kl_div = kl_div.sum(dim=-1)  # (B,)
        
        # Apply mask for conflict resolution
        if mask is not None:
            kl_div = kl_div * mask
            loss = kl_div.sum() / (mask.sum() + 1e-8)
        else:
            loss = kl_div.mean()
        
        return loss


class PTGLoss(nn.Module):
    """
    Complete PTG loss function.
    
    Phase 1: L_base = L_action + λ_dur L_duration + λ_obj L_object + λ_goal L_goal
    Phase 2: L_total = L_base + λ_gram L_grammar
    """
    
    def __init__(
        self,
        lambda_dur: float = 1.0,
        lambda_obj: float = 1.0,
        lambda_goal: float = 1.0,
        lambda_gram: float = 0.5,
        use_grammar: bool = False,
        use_goal: bool = False
    ):
        """
        Args:
            lambda_dur: Weight for duration loss
            lambda_obj: Weight for object loss
            lambda_goal: Weight for goal loss
            lambda_gram: Weight for grammar KL loss
            use_grammar: Enable grammar regularization (Phase 2)
            use_goal: Enable goal prediction
        """
        super().__init__()
        
        self.lambda_dur = lambda_dur
        self.lambda_obj = lambda_obj
        self.lambda_goal = lambda_goal
        self.lambda_gram = lambda_gram
        self.use_grammar = use_grammar
        self.use_goal = use_goal
        
        # Loss components
        self.action_loss = ActionLoss()
        self.duration_loss = DurationLoss()
        self.object_loss = ObjectLoss()
        self.goal_loss = GoalLoss() if use_goal else None
        self.grammar_kl_loss = GrammarKLLoss() if use_grammar else None
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        grammar_probs: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.
        
        Args:
            predictions: Dictionary with model outputs
            targets: Dictionary with ground truth
            grammar_probs: (B, num_actions) grammar probabilities for Phase 2
            
        Returns:
            Dictionary with loss components and total loss
        """
        losses = {}
        
        # Action loss
        losses['action'] = self.action_loss(
            predictions['action_logits'],
            targets['actions']
        )
        
        # Duration loss
        losses['duration'] = self.duration_loss(
            predictions['duration_pred'],
            targets['durations'],
            mask=targets.get('duration_mask')
        )
        
        # Object loss
        losses['object'] = self.object_loss(
            predictions['object_logits'],
            targets['objects']
        )
        
        # Goal loss (optional)
        if self.use_goal and 'goal_logits' in predictions:
            losses['goal'] = self.goal_loss(
                predictions['goal_logits'],
                targets['goals']
            )
        
        # Base loss (Phase 1)
        base_loss = (
            losses['action'] +
            self.lambda_dur * losses['duration'] +
            self.lambda_obj * losses['object']
        )
        
        if self.use_goal and 'goal' in losses:
            base_loss += self.lambda_goal * losses['goal']
        
        losses['base'] = base_loss
        
        # Grammar KL loss (Phase 2)
        if self.use_grammar and grammar_probs is not None:
            losses['grammar_kl'] = self.grammar_kl_loss(
                predictions['action_logits'],
                grammar_probs,
                mask=targets.get('grammar_mask')
            )
            
            total_loss = base_loss + self.lambda_gram * losses['grammar_kl']
        else:
            total_loss = base_loss
        
        losses['total'] = total_loss
        
        return losses


# Example usage
if __name__ == '__main__':
    # Create loss function
    criterion = PTGLoss(
        lambda_dur=1.0,
        lambda_obj=1.0,
        lambda_goal=1.0,
        lambda_gram=0.5,
        use_grammar=True,
        use_goal=True
    )
    
    # Mock predictions
    B, num_actions, num_objects, num_goals = 4, 466, 16, 8
    predictions = {
        'action_logits': torch.randn(B, num_actions),
        'duration_pred': torch.randn(B),
        'object_logits': torch.randn(B, num_objects),
        'goal_logits': torch.randn(B, num_goals)
    }
    
    # Mock targets
    targets = {
        'actions': torch.randint(0, num_actions, (B,)),
        'durations': torch.randn(B).abs() * 20 + 10,
        'objects': torch.randint(0, 2, (B, num_objects)).float(),
        'goals': torch.randint(0, num_goals, (B,))
    }
    
    # Mock grammar probabilities
    grammar_probs = F.softmax(torch.randn(B, num_actions), dim=-1)
    
    # Compute loss
    losses = criterion(predictions, targets, grammar_probs)
    
    print("Loss components:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")

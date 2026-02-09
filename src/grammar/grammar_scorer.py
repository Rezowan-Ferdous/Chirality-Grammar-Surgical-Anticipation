"""
Grammar scoring for Probabilistic Temporal Grammar.

Computes log-scores for action candidates integrating multiple knowledge sources:
transition probabilities, object consistency, duration plausibility, chirality priors,
and goal orientation.

Paper: Equation in Section 3.2
"""
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F

from .transition_model import TransitionModel
from .duration_model import DurationModel
from .chirality_lexicon import ChiralityLexicon


class GrammarScorer:
    """
    Grammar scoring function for PTG.
    
    Paper: "s_G(a) = log P(a|h) + λ_o log⟨Ô, O(a)⟩ + λ_d log L_dur(d̂|a) 
                     + log γ(a_{t-1}, a) + λ_r R_goal(h, a; G, R)"
    """
    
    def __init__(
        self,
        transition_model: TransitionModel,
        duration_model: DurationModel,
        chirality_lexicon: ChiralityLexicon,
        object_model: Optional[Dict] = None,
        goal_model: Optional[Dict] = None,
        lambda_o: float = 1.0,
        lambda_d: float = 1.0,
        lambda_r: float = 0.5,
        temperature: float = 1.0
    ):
        """
        Args:
            transition_model: Markov transition probabilities
            duration_model: Duration statistics
            chirality_lexicon: Chiral action pairs
            object_model: Object transition statistics (optional)
            goal_model: Goal matrices G, R (optional)
            lambda_o: Weight for object consistency
            lambda_d: Weight for duration plausibility
            lambda_r: Weight for goal orientation
            temperature: Temperature for softmax
        """
        self.transition_model = transition_model
        self.duration_model = duration_model
        self.chirality_lexicon = chirality_lexicon
        self.object_model = object_model or {}
        self.goal_model = goal_model or {}
        
        self.lambda_o = lambda_o
        self.lambda_d = lambda_d
        self.lambda_r = lambda_r
        self.temperature = temperature
    
    def score_action(
        self,
        action: str,
        history: Tuple[str, ...],
        predicted_duration: float = None,
        predicted_objects: np.ndarray = None
    ) -> float:
        """
        Compute grammar log-score for a single action.
        
        Args:
            action: Candidate action
            history: Previous actions (up to markov_order)
            predicted_duration: Predicted duration from neural model
            predicted_objects: Predicted object state vector from neural model
            
        Returns:
            Log-score s_G(a)
        """
        score = 0.0
        
        # 1. Transition probability: log P(a | h)
        trans_prob = self.transition_model.get_transition_prob(history, action)
        score += np.log(max(trans_prob, 1e-10))
        
        # 2. Object consistency: λ_o log⟨Ô, O(a)⟩
        if predicted_objects is not None and action in self.object_model:
            expected_objects = self.object_model[action]
            consistency = np.dot(predicted_objects, expected_objects)
            score += self.lambda_o * np.log(max(consistency, 1e-10))
        
        # 3. Duration plausibility: λ_d log L_dur(d̂ | a)
        if predicted_duration is not None:
            duration_ll = self.duration_model.log_likelihood(action, predicted_duration)
            score += self.lambda_d * duration_ll
        
        # 4. Chirality prior: log γ(a_{t-1}, a)
        if len(history) > 0:
            prev_action = history[-1]
            chirality_prior = self.chirality_lexicon.get_chirality_prior(prev_action, action)
            score += np.log(chirality_prior)
        
        # 5. Goal orientation: λ_r R_goal(h, a; G, R)
        if self.goal_model:
            goal_reward = self._compute_goal_reward(history, action)
            score += self.lambda_r * goal_reward
        
        return score
    
    def _compute_goal_reward(
        self,
        history: Tuple[str, ...],
        action: str
    ) -> float:
        """
        Compute goal-oriented reward.
        
        Paper: "R_goal provides a soft reward bias that reflects goal reachability"
        """
        # Placeholder for GoMMC-based goal rewards
        # In full implementation, this would use learned reward matrices R^{jk}
        if 'R' in self.goal_model and action in self.goal_model['R']:
            return self.goal_model['R'][action]
        return 0.0
    
    def get_distribution(
        self,
        history: Tuple[str, ...],
        predicted_duration: float = None,
        predicted_objects: np.ndarray = None,
        return_scores: bool = False
    ) -> Dict[str, float]:
        """
        Get grammar probability distribution over all actions.
        
        Paper: "P_G(a) = exp(s_G(a)/τ) / Σ exp(s_G(a')/τ)"
        
        Args:
            history: Previous actions
            predicted_duration: Predicted duration
            predicted_objects: Predicted object states
            return_scores: If True, return raw scores instead of probabilities
            
        Returns:
            Dictionary mapping actions to probabilities (or scores)
        """
        scores = {}
        
        # Compute scores for all actions in vocabulary
        for action in self.transition_model.vocabulary:
            score = self.score_action(
                action, history, predicted_duration, predicted_objects
            )
            scores[action] = score
        
        if return_scores:
            return scores
        
        # Apply temperature and softmax
        actions = list(scores.keys())
        score_values = np.array([scores[a] for a in actions])
        
        # Temperature scaling
        score_values = score_values / self.temperature
        
        # Softmax
        exp_scores = np.exp(score_values - np.max(score_values))  # Numerical stability
        probs = exp_scores / exp_scores.sum()
        
        return {action: prob for action, prob in zip(actions, probs)}
    
    def get_distribution_torch(
        self,
        history: Tuple[str, ...],
        predicted_duration: torch.Tensor = None,
        predicted_objects: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Get grammar distribution as PyTorch tensor for KL divergence loss.
        
        Args:
            history: Previous actions
            predicted_duration: (B,) tensor
            predicted_objects: (B, num_objects) tensor
            
        Returns:
            (B, num_actions) probability tensor
        """
        batch_size = 1 if predicted_duration is None else predicted_duration.shape[0]
        num_actions = len(self.transition_model.vocabulary)
        action_to_idx = {a: i for i, a in enumerate(sorted(self.transition_model.vocabulary))}
        
        probs = torch.zeros(batch_size, num_actions)
        
        for b in range(batch_size):
            dur = predicted_duration[b].item() if predicted_duration is not None else None
            obj = predicted_objects[b].cpu().numpy() if predicted_objects is not None else None
            
            dist = self.get_distribution(history, dur, obj)
            
            for action, prob in dist.items():
                probs[b, action_to_idx[action]] = prob
        
        return probs


# Example usage
if __name__ == '__main__':
    from .transition_model import TransitionModel
    from .duration_model import DurationModel
    from .chirality_lexicon import ChiralityLexicon
    
    # Create sample models
    sequences = [
        ['pick', 'grasp', 'pull', 'cut', 'drop'],
        ['pick', 'grasp', 'push', 'cut', 'drop'],
    ]
    segments = [
        ('pick', 0, 10),
        ('grasp', 11, 50),
        ('pull', 51, 70),
        ('push', 51, 75),
        ('cut', 71, 90),
        ('drop', 91, 100),
    ]
    
    transition_model = TransitionModel(markov_order=2)
    transition_model.fit(sequences)
    
    duration_model = DurationModel()
    duration_model.fit(segments)
    
    chirality_lexicon = ChiralityLexicon(prior_boost=1.2)
    
    # Create scorer
    scorer = GrammarScorer(
        transition_model=transition_model,
        duration_model=duration_model,
        chirality_lexicon=chirality_lexicon,
        lambda_o=1.0,
        lambda_d=1.0,
        lambda_r=0.5,
        temperature=1.0
    )
    
    # Test scoring
    history = ('pick', 'grasp')
    print("Grammar scores:")
    scores = scorer.get_distribution(history, predicted_duration=20.0, return_scores=True)
    for action, score in sorted(scores.items(), key=lambda x: -x[1])[:5]:
        print(f"  {action}: {score:.4f}")
    
    # Test probability distribution
    print("\nGrammar probability distribution:")
    dist = scorer.get_distribution(history, predicted_duration=20.0)
    for action, prob in sorted(dist.items(), key=lambda x: -x[1])[:5]:
        print(f"  {action}: {prob:.4f}")

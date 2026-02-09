"""
Transition model for Probabilistic Temporal Grammar.

Estimates N-th order Markov transition probabilities from action sequences.

Paper: "Estimate N-th order transition probabilities P(a_t|a_{t-N:t-1})"
"""
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
import numpy as np
import pickle


class TransitionModel:
    """
    N-th order Markov transition model.
    
    Paper: "After hierarchical abstraction, the induction computes the 
    N-th order transition counts and normalizes them to produce P 
    (Markovian transition probabilities)"
    """
    
    def __init__(self, markov_order: int = 2, smoothing: float = 1e-5):
        """
        Args:
            markov_order: Order of Markov model (paper uses 2)
            smoothing: Laplace smoothing parameter
        """
        self.markov_order = markov_order
        self.smoothing = smoothing
        
        # Transition counts: history -> {next_action: count}
        self.transition_counts: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        
        # Transition probabilities: history -> {next_action: prob}
        self.transition_probs: Dict[Tuple[str, ...], Dict[str, float]] = {}
        
        # All observed actions (vocabulary)
        self.vocabulary: set = set()
    
    def fit(self, sequences: List[List[str]]):
        """
        Learn transition probabilities from sequences.
        
        Args:
            sequences: List of action sequences (after hierarchical abstraction)
        """
        # Count transitions
        for sequence in sequences:
            for i in range(len(sequence)):
                # Get history (previous N actions)
                start_idx = max(0, i - self.markov_order)
                history = tuple(sequence[start_idx:i])
                
                # Current action
                action = sequence[i]
                
                # Update counts
                self.transition_counts[history][action] += 1
                self.vocabulary.add(action)
        
        # Normalize to probabilities with Laplace smoothing
        vocab_size = len(self.vocabulary)
        
        for history, action_counts in self.transition_counts.items():
            total_count = sum(action_counts.values())
            
            # Add smoothing
            total_with_smoothing = total_count + self.smoothing * vocab_size
            
            self.transition_probs[history] = {
                action: (count + self.smoothing) / total_with_smoothing
                for action, count in action_counts.items()
            }
            
            # Add unseen actions with smoothing probability
            for action in self.vocabulary:
                if action not in self.transition_probs[history]:
                    self.transition_probs[history][action] = self.smoothing / total_with_smoothing
    
    def get_transition_prob(
        self,
        history: Tuple[str, ...],
        action: str
    ) -> float:
        """
        Get P(action | history).
        
        Args:
            history: Tuple of previous actions (length <= markov_order)
            action: Next action
            
        Returns:
            Transition probability
        """
        # Truncate history to markov_order
        if len(history) > self.markov_order:
            history = history[-self.markov_order:]
        
        if history in self.transition_probs:
            return self.transition_probs[history].get(action, self.smoothing)
        else:
            # Unseen history, use uniform distribution
            return 1.0 / len(self.vocabulary) if self.vocabulary else 0.0
    
    def get_distribution(
        self,
        history: Tuple[str, ...]
    ) -> Dict[str, float]:
        """
        Get full probability distribution P(· | history).
        
        Args:
            history: Tuple of previous actions
            
        Returns:
            Dictionary mapping actions to probabilities
        """
        # Truncate history
        if len(history) > self.markov_order:
            history = history[-self.markov_order:]
        
        if history in self.transition_probs:
            return self.transition_probs[history]
        else:
            # Unseen history, uniform
            uniform_prob = 1.0 / len(self.vocabulary)
            return {action: uniform_prob for action in self.vocabulary}
    
    def sample_next(
        self,
        history: Tuple[str, ...],
        temperature: float = 1.0
    ) -> str:
        """
        Sample next action from P(· | history).
        
        Args:
            history: Previous actions
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            Sampled action
        """
        dist = self.get_distribution(history)
        
        # Apply temperature
        if temperature != 1.0:
            actions = list(dist.keys())
            probs = np.array([dist[a] for a in actions])
            probs = probs ** (1.0 / temperature)
            probs = probs / probs.sum()
            dist = {a: p for a, p in zip(actions, probs)}
        
        # Sample
        actions = list(dist.keys())
        probs = [dist[a] for a in actions]
        
        return np.random.choice(actions, p=probs)
    
    def get_statistics(self) -> Dict:
        """Get model statistics."""
        num_histories = len(self.transition_probs)
        avg_branching = np.mean([
            len(actions) for actions in self.transition_probs.values()
        ]) if self.transition_probs else 0
        
        return {
            'vocabulary_size': len(self.vocabulary),
            'num_histories': num_histories,
            'avg_branching_factor': avg_branching,
            'markov_order': self.markov_order
        }
    
    def save(self, filepath: str):
        """Save model to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'markov_order': self.markov_order,
                'smoothing': self.smoothing,
                'transition_counts': dict(self.transition_counts),
                'transition_probs': self.transition_probs,
                'vocabulary': self.vocabulary
            }, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'TransitionModel':
        """Load model from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(
            markov_order=data['markov_order'],
            smoothing=data['smoothing']
        )
        model.transition_counts = defaultdict(Counter, data['transition_counts'])
        model.transition_probs = data['transition_probs']
        model.vocabulary = data['vocabulary']
        
        return model


# Example usage
if __name__ == '__main__':
    # Sample sequences
    sequences = [
        ['C0', 'C1', 'C2', 'cut', 'drop'],
        ['C0', 'C1', 'push', 'cut', 'drop'],
        ['C0', 'C1', 'C2', 'cut', 'suture'],
        ['pick', 'C1', 'C2', 'cut'],
    ]
    
    # Train model
    model = TransitionModel(markov_order=2)
    model.fit(sequences)
    
    # Get statistics
    stats = model.get_statistics()
    print("Transition model statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test transition probabilities
    print("\nTransition probabilities:")
    history = ('C0', 'C1')
    dist = model.get_distribution(history)
    print(f"P(· | {history}):")
    for action, prob in sorted(dist.items(), key=lambda x: -x[1])[:5]:
        print(f"  {action}: {prob:.4f}")
    
    # Sample next action
    print("\nSampled next actions:")
    for _ in range(5):
        next_action = model.sample_next(history)
        print(f"  {next_action}")

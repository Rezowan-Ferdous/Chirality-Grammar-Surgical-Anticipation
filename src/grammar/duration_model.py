"""
Duration model for Probabilistic Temporal Grammar.

Computes robust duration statistics for each action using median + MAD.

Paper: "robust duration statistics for each state to populate D 
(median, MAD, sample standard deviation)"
"""
from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np
import pickle


class DurationModel:
    """
    Robust duration statistics for actions.
    
    Paper: "Collect durations for each state, compute robust statistics D"
    Uses median and Median Absolute Deviation (MAD) for robustness to outliers.
    """
    
    def __init__(self):
        """Initialize duration model."""
        # Raw duration observations: action -> list of durations
        self.duration_observations: Dict[str, List[float]] = defaultdict(list)
        
        # Robust statistics: action -> {median, MAD, std, min, max}
        self.duration_stats: Dict[str, Dict[str, float]] = {}
    
    def fit(
        self,
        action_segments: List[Tuple[str, int, int]]
    ):
        """
        Learn duration statistics from action segments.
        
        Args:
            action_segments: List of (action, start_frame, end_frame) tuples
        """
        # Collect durations
        for action, start, end in action_segments:
            duration = end - start + 1  # Inclusive
            self.duration_observations[action].append(duration)
        
        # Compute robust statistics for each action
        for action, durations in self.duration_observations.items():
            durations = np.array(durations)
            
            # Median
            median = np.median(durations)
            
            # Median Absolute Deviation (MAD)
            mad = np.median(np.abs(durations - median))
            
            # Sample standard deviation
            std = np.std(durations, ddof=1) if len(durations) > 1 else 0.0
            
            # Min and max
            min_dur = np.min(durations)
            max_dur = np.max(durations)
            
            self.duration_stats[action] = {
                'median': float(median),
                'mad': float(mad),
                'std': float(std),
                'min': float(min_dur),
                'max': float(max_dur),
                'count': len(durations)
            }
    
    def get_stats(self, action: str) -> Dict[str, float]:
        """Get duration statistics for an action."""
        return self.duration_stats.get(action, {
            'median': 0.0,
            'mad': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'count': 0
        })
    
    def log_likelihood(
        self,
        action: str,
        duration: float,
        use_laplace: bool = True
    ) -> float:
        """
        Compute log-likelihood of a duration given an action.
        
        Paper: "λ_d log L_dur(d̂ | a)"
        
        Uses a robust likelihood based on Laplace distribution with
        location=median and scale=MAD.
        
        Args:
            action: Action label
            duration: Observed duration
            use_laplace: If True, use Laplace distribution, else Gaussian
            
        Returns:
            Log-likelihood
        """
        stats = self.get_stats(action)
        
        if stats['count'] == 0:
            # Unseen action, return neutral log-likelihood
            return 0.0
        
        median = stats['median']
        mad = max(stats['mad'], 1e-6)  # Avoid division by zero
        
        if use_laplace:
            # Laplace distribution: L(x|μ,b) = (1/2b) exp(-|x-μ|/b)
            # log L = -log(2b) - |x-μ|/b
            log_likelihood = -np.log(2 * mad) - np.abs(duration - median) / mad
        else:
            # Gaussian distribution with std
            std = max(stats['std'], 1e-6)
            log_likelihood = -0.5 * np.log(2 * np.pi * std**2) - 0.5 * ((duration - median) / std)**2
        
        return log_likelihood
    
    def is_plausible(
        self,
        action: str,
        duration: float,
        n_mads: float = 3.0
    ) -> bool:
        """
        Check if a duration is plausible for an action.
        
        Args:
            action: Action label
            duration: Duration to check
            n_mads: Number of MADs from median for plausibility
            
        Returns:
            True if duration is within n_mads of median
        """
        stats = self.get_stats(action)
        
        if stats['count'] == 0:
            return True  # Unknown action, assume plausible
        
        median = stats['median']
        mad = stats['mad']
        
        deviation = np.abs(duration - median)
        return deviation <= n_mads * mad
    
    def get_summary(self) -> Dict:
        """Get summary statistics across all actions."""
        all_medians = [stats['median'] for stats in self.duration_stats.values()]
        all_mads = [stats['mad'] for stats in self.duration_stats.values()]
        
        return {
            'num_actions': len(self.duration_stats),
            'median_duration_mean': np.mean(all_medians) if all_medians else 0.0,
            'median_duration_std': np.std(all_medians) if all_medians else 0.0,
            'mad_mean': np.mean(all_mads) if all_mads else 0.0,
            'mad_std': np.std(all_mads) if all_mads else 0.0
        }
    
    def save(self, filepath: str):
        """Save model to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'duration_observations': dict(self.duration_observations),
                'duration_stats': self.duration_stats
            }, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'DurationModel':
        """Load model from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        model = cls()
        model.duration_observations = defaultdict(list, data['duration_observations'])
        model.duration_stats = data['duration_stats']
        
        return model


# Example usage
if __name__ == '__main__':
    # Sample action segments (action, start_frame, end_frame)
    segments = [
        ('pick', 0, 10),
        ('pick', 100, 112),
        ('grasp', 11, 50),
        ('grasp', 113, 145),
        ('cut', 51, 70),
        ('cut', 146, 168),
        ('cut', 200, 215),
    ]
    
    # Train model
    model = DurationModel()
    model.fit(segments)
    
    # Get statistics
    print("Duration statistics:")
    for action in ['pick', 'grasp', 'cut']:
        stats = model.get_stats(action)
        print(f"\n{action}:")
        print(f"  Median: {stats['median']:.1f} frames")
        print(f"  MAD: {stats['mad']:.1f} frames")
        print(f"  Std: {stats['std']:.1f} frames")
        print(f"  Range: [{stats['min']:.0f}, {stats['max']:.0f}]")
        print(f"  Observations: {stats['count']}")
    
    # Test log-likelihood
    print("\nLog-likelihoods:")
    print(f"pick @ 10 frames: {model.log_likelihood('pick', 10):.4f}")
    print(f"pick @ 50 frames: {model.log_likelihood('pick', 50):.4f}")
    
    # Test plausibility
    print("\nPlausibility tests:")
    print(f"pick @ 10 frames: {model.is_plausible('pick', 10)}")
    print(f"pick @ 100 frames: {model.is_plausible('pick', 100)}")
    
    # Summary
    summary = model.get_summary()
    print("\nOverall summary:")
    for key, value in summary.items():
        print(f"  {key}: {value:.2f}")

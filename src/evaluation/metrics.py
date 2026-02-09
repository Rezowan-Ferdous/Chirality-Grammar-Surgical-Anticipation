"""
Evaluation metrics for PTG.

Implements:
- Mean over Classes (MoC) accuracy
- Per-horizon metrics
- Chirality F1 score
- Edit distance
- Segmental metrics

Paper: "We evaluate on MoC accuracy at different prediction horizons
(τ ∈ {1, 5, 10, 20, 50}) and compute chirality-specific F1"
"""
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from collections import defaultdict


class PTGEvaluator:
    """Comprehensive evaluation for PTG model."""
    
    def __init__(
        self,
        action_vocab: List[str],
        chiral_pairs: List[Tuple[str, str]],
        eval_horizons: List[int] = [1, 5, 10, 20, 50]
    ):
        """
        Args:
            action_vocab: List of all action labels
            chiral_pairs: List of (action1, action2) chiral pairs
            eval_horizons: Horizons to evaluate
        """
        self.action_vocab = action_vocab
        self.num_actions = len(action_vocab)
        self.action_to_idx = {a: i for i, a in enumerate(action_vocab)}
        self.chiral_pairs = chiral_pairs
        self.eval_horizons = eval_horizons
        
        # Identify chiral actions
        self.chiral_actions = set()
        for a1, a2 in chiral_pairs:
            self.chiral_actions.add(a1)
            self.chiral_actions.add(a2)
        
        # Reset accumulators
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = defaultdict(list)
        self.targets = defaultdict(list)
        self.durations_pred = []
        self.durations_true = []
        self.edit_distances = []
    
    def update(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        horizon: int
    ):
        """
        Update metrics with batch predictions.
        
        Args:
            predictions: Model outputs
            targets: Ground truth
            horizon: Prediction horizon for this batch
        """
        # Action predictions
        if 'action_logits' in predictions:
            pred_actions = predictions['action_logits'].argmax(dim=-1)  # (B, H)
            true_actions = targets['actions']  # (B, H)
            
            # Store per-horizon
            for h in range(min(horizon, pred_actions.shape[1])):
                self.predictions[h + 1].extend(pred_actions[:, h].cpu().numpy())
                self.targets[h + 1].extend(true_actions[:, h].cpu().numpy())
        
        # Duration predictions
        if 'duration_pred' in predictions:
            self.durations_pred.extend(predictions['duration_pred'].cpu().numpy().flatten())
            self.durations_true.extend(targets['durations'].cpu().numpy().flatten())
    
    def compute_moc_accuracy(self) -> Dict[int, float]:
        """
        Compute Mean over Classes (MoC) accuracy.
        
        Paper: "MoC = (1/|C|) Σ_c (TP_c / (TP_c + FN_c))"
        
        Returns:
            Dictionary mapping horizon to MoC accuracy
        """
        moc_scores = {}
        
        for horizon in self.eval_horizons:
            if horizon not in self.predictions:
                continue
            
            preds = np.array(self.predictions[horizon])
            targets = np.array(self.targets[horizon])
            
            # Per-class recall
            class_recalls = []
            for class_idx in range(self.num_actions):
                # True positives
                tp = np.sum((preds == class_idx) & (targets == class_idx))
                # False negatives
                fn = np.sum((preds != class_idx) & (targets == class_idx))
                
                if (tp + fn) > 0:
                    recall = tp / (tp + fn)
                    class_recalls.append(recall)
            
            # Mean over classes
            moc_scores[horizon] = np.mean(class_recalls) if class_recalls else 0.0
        
        return moc_scores
    
    def compute_accuracy(self) -> Dict[int, float]:
        """Compute standard accuracy per horizon."""
        acc_scores = {}
        
        for horizon in self.eval_horizons:
            if horizon not in self.predictions:
                continue
            
            preds = np.array(self.predictions[horizon])
            targets = np.array(self.targets[horizon])
            
            # Filter out ignore index (-100)
            valid_mask = targets != -100
            if valid_mask.sum() > 0:
                acc_scores[horizon] = accuracy_score(
                    targets[valid_mask],
                    preds[valid_mask]
                )
        
        return acc_scores
    
    def compute_chirality_metrics(self) -> Dict[str, float]:
        """
        Compute chirality-specific metrics.
        
        Returns:
            Dictionary with chirality precision, recall, F1
        """
        # Get all predictions and targets
        all_preds = []
        all_targets = []
        
        for horizon in self.predictions.keys():
            all_preds.extend(self.predictions[horizon])
            all_targets.extend(self.targets[horizon])
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Create binary labels: 1 if chiral, 0 otherwise
        chiral_indices = [self.action_to_idx[a] for a in self.chiral_actions 
                         if a in self.action_to_idx]
        
        pred_is_chiral = np.isin(all_preds, chiral_indices).astype(int)
        target_is_chiral = np.isin(all_targets, chiral_indices).astype(int)
        
        # Filter out ignore index
        valid_mask = all_targets != -100
        
        if valid_mask.sum() > 0:
            precision, recall, f1, _ = precision_recall_fscore_support(
                target_is_chiral[valid_mask],
                pred_is_chiral[valid_mask],
                average='binary',
                zero_division=0
            )
            
            return {
                'chirality_precision': precision,
                'chirality_recall': recall,
                'chirality_f1': f1
            }
        
        return {
            'chirality_precision': 0.0,
            'chirality_recall': 0.0,
            'chirality_f1': 0.0
        }
    
    def compute_duration_metrics(self) -> Dict[str, float]:
        """Compute duration prediction metrics."""
        if not self.durations_pred:
            return {}
        
        preds = np.array(self.durations_pred)
        targets = np.array(self.durations_true)
        
        # Filter out invalid values
        valid_mask = (targets > 0) & ~np.isnan(preds)
        
        if valid_mask.sum() > 0:
            preds = preds[valid_mask]
            targets = targets[valid_mask]
            
            # MAE
            mae = np.mean(np.abs(preds - targets))
            
            # RMSE
            rmse = np.sqrt(np.mean((preds - targets) ** 2))
            
            # Relative error
            rel_error = np.mean(np.abs(preds - targets) / (targets + 1e-8))
            
            return {
                'duration_mae': mae,
                'duration_rmse': rmse,
                'duration_rel_error': rel_error
            }
        
        return {}
    
    def compute_edit_distance(
        self,
        pred_sequence: List[int],
        target_sequence: List[int]
    ) -> int:
        """
        Compute Levenshtein edit distance between sequences.
        
        Args:
            pred_sequence: Predicted action sequence
            target_sequence: Ground truth sequence
            
        Returns:
            Edit distance
        """
        m, n = len(pred_sequence), len(target_sequence)
        
        # DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_sequence[i - 1] == target_sequence[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # Deletion
                        dp[i][j - 1],      # Insertion
                        dp[i - 1][j - 1]   # Substitution
                    )
        
        return dp[m][n]
    
    def compute_normalized_edit_distance(
        self,
        pred_sequence: List[int],
        target_sequence: List[int]
    ) -> float:
        """Compute normalized edit distance (0-1 range)."""
        ed = self.compute_edit_distance(pred_sequence, target_sequence)
        max_len = max(len(pred_sequence), len(target_sequence))
        return ed / max_len if max_len > 0 else 0.0
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary of all computed metrics."""
        summary = {}
        
        # MoC accuracy
        moc_scores = self.compute_moc_accuracy()
        for horizon, score in moc_scores.items():
            summary[f'moc_acc_h{horizon}'] = score
        
        # Standard accuracy
        acc_scores = self.compute_accuracy()
        for horizon, score in acc_scores.items():
            summary[f'acc_h{horizon}'] = score
        
        # Chirality metrics
        chiral_metrics = self.compute_chirality_metrics()
        summary.update(chiral_metrics)
        
        # Duration metrics
        duration_metrics = self.compute_duration_metrics()
        summary.update(duration_metrics)
        
        return summary
    
    def print_summary(self):
        """Print formatted summary."""
        summary = self.get_summary()
        
        print("=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        
        # MoC Accuracy
        print("\nMean over Classes (MoC) Accuracy:")
        for horizon in self.eval_horizons:
            key = f'moc_acc_h{horizon}'
            if key in summary:
                print(f"  Horizon {horizon:2d}: {summary[key]:.4f}")
        
        # Standard Accuracy
        print("\nStandard Accuracy:")
        for horizon in self.eval_horizons:
            key = f'acc_h{horizon}'
            if key in summary:
                print(f"  Horizon {horizon:2d}: {summary[key]:.4f}")
        
        # Chirality
        print("\nChirality Metrics:")
        for key in ['chirality_precision', 'chirality_recall', 'chirality_f1']:
            if key in summary:
                print(f"  {key.replace('_', ' ').title()}: {summary[key]:.4f}")
        
        # Duration
        print("\nDuration Metrics:")
        for key in ['duration_mae', 'duration_rmse', 'duration_rel_error']:
            if key in summary:
                print(f"  {key.replace('_', ' ').upper()}: {summary[key]:.4f}")
        
        print("=" * 60)


# Example usage
if __name__ == '__main__':
    # Create evaluator
    action_vocab = ['pick', 'grasp', 'pull', 'push', 'cut', 'drop']
    chiral_pairs = [('pick', 'drop'), ('pull', 'push')]
    
    evaluator = PTGEvaluator(
        action_vocab=action_vocab,
        chiral_pairs=chiral_pairs,
        eval_horizons=[1, 5, 10]
    )
    
    # Mock predictions
    predictions = {
        'action_logits': torch.randn(4, 10, 6),  # (B, H, C)
        'duration_pred': torch.randn(4, 10).abs() * 20
    }
    
    targets = {
        'actions': torch.randint(0, 6, (4, 10)),
        'durations': torch.randn(4, 10).abs() * 20
    }
    
    # Update metrics
    evaluator.update(predictions, targets, horizon=10)
    
    # Print summary
    evaluator.print_summary()

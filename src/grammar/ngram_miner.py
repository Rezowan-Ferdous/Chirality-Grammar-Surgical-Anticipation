"""
N-gram mining for Probabilistic Temporal Grammar (PTG).

Extracts frequent contiguous subsequences from surgical action sequences.
Implements bottom-up hierarchical pattern discovery as described in the paper.
"""
from typing import List, Dict, Tuple, Set
from collections import Counter, defaultdict
from dataclasses import dataclass
import numpy as np


@dataclass
class NGram:
    """Represents an n-gram pattern."""
    pattern: Tuple[str, ...]  # Sequence of action symbols
    frequency: int
    positions: List[int]  # Positions where this n-gram occurs


class NGramMiner:
    """
    Mines frequent n-grams from action sequences.
    
    Paper: "The induction algorithm proceeds bottom-up, iteratively mining 
    frequent contiguous subsequences (n-grams), abstract them into unique 
    composite tokens."
    """
    
    def __init__(
        self,
        min_freq: int = 3,
        max_ngram_size: int = 4
    ):
        """
        Args:
            min_freq: Minimum frequency threshold for n-grams
            max_ngram_size: Maximum n-gram size (paper uses 4)
        """
        self.min_freq = min_freq
        self.max_ngram_size = max_ngram_size
        
    def mine_ngrams(
        self,
        sequences: List[List[str]],
        n: int
    ) -> Dict[Tuple[str, ...], NGram]:
        """
        Mine all n-grams of size n from sequences.
        
        Args:
            sequences: List of action sequences
            n: N-gram size
            
        Returns:
            Dictionary mapping n-gram patterns to NGram objects
        """
        ngram_counts = Counter()
        ngram_positions = defaultdict(list)
        
        for seq_idx, sequence in enumerate(sequences):
            for i in range(len(sequence) - n + 1):
                pattern = tuple(sequence[i:i+n])
                ngram_counts[pattern] += 1
                ngram_positions[pattern].append((seq_idx, i))
        
        # Filter by frequency
        frequent_ngrams = {}
        for pattern, count in ngram_counts.items():
            if count >= self.min_freq:
                frequent_ngrams[pattern] = NGram(
                    pattern=pattern,
                    frequency=count,
                    positions=ngram_positions[pattern]
                )
        
        return frequent_ngrams
    
    def mine_all_ngrams(
        self,
        sequences: List[List[str]]
    ) -> Dict[int, Dict[Tuple[str, ...], NGram]]:
        """
        Mine n-grams for all sizes from 2 to max_ngram_size.
        
        Args:
            sequences: List of action sequences
            
        Returns:
            Dictionary mapping n-gram size to frequent n-grams
        """
        all_ngrams = {}
        
        for n in range(2, self.max_ngram_size + 1):
            ngrams = self.mine_ngrams(sequences, n)
            if ngrams:
                all_ngrams[n] = ngrams
                print(f"Found {len(ngrams)} frequent {n}-grams")
        
        return all_ngrams
    
    def get_statistics(
        self,
        all_ngrams: Dict[int, Dict[Tuple[str, ...], NGram]]
    ) -> Dict[str, int]:
        """Get statistics about mined n-grams."""
        stats = {}
        total_patterns = 0
        
        for n, ngrams in all_ngrams.items():
            stats[f'{n}-grams'] = len(ngrams)
            total_patterns += len(ngrams)
        
        stats['total_patterns'] = total_patterns
        return stats


class GreedyMatcher:
    """
    Greedy left-to-right maximal matching for pattern replacement.
    
    Paper: "replace occurrences in the corpus by maximal, left-to-right 
    greedy matching (longer patterns first)"
    """
    
    def __init__(self, patterns: Dict[Tuple[str, ...], str]):
        """
        Args:
            patterns: Mapping from n-gram patterns to composite symbols
        """
        # Sort patterns by length (longest first) for greedy matching
        self.patterns = sorted(
            patterns.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
    
    def replace_in_sequence(
        self,
        sequence: List[str]
    ) -> List[str]:
        """
        Replace n-grams with composite symbols using greedy matching.
        
        Args:
            sequence: Original action sequence
            
        Returns:
            New sequence with patterns replaced by composite symbols
        """
        result = []
        i = 0
        
        while i < len(sequence):
            matched = False
            
            # Try to match longest pattern first
            for pattern, composite in self.patterns:
                pattern_len = len(pattern)
                
                if i + pattern_len <= len(sequence):
                    # Check if pattern matches at position i
                    if tuple(sequence[i:i+pattern_len]) == pattern:
                        result.append(composite)
                        i += pattern_len
                        matched = True
                        break
            
            # No pattern matched, keep original symbol
            if not matched:
                result.append(sequence[i])
                i += 1
        
        return result
    
    def replace_in_sequences(
        self,
        sequences: List[List[str]]
    ) -> List[List[str]]:
        """Replace patterns in all sequences."""
        return [self.replace_in_sequence(seq) for seq in sequences]


def compute_ngram_coverage(
    original_sequences: List[List[str]],
    replaced_sequences: List[List[str]],
    composite_symbols: Set[str]
) -> Dict[str, float]:
    """
    Compute coverage statistics after pattern replacement.
    
    Args:
        original_sequences: Original sequences
        replaced_sequences: Sequences after replacement
        composite_symbols: Set of composite symbols
        
    Returns:
        Dictionary with coverage metrics
    """
    total_original_length = sum(len(seq) for seq in original_sequences)
    total_replaced_length = sum(len(seq) for seq in replaced_sequences)
    
    # Count composite symbols
    composite_count = sum(
        sum(1 for token in seq if token in composite_symbols)
        for seq in replaced_sequences
    )
    
    compression_ratio = total_replaced_length / total_original_length
    composite_ratio = composite_count / total_replaced_length
    
    return {
        'original_length': total_original_length,
        'replaced_length': total_replaced_length,
        'compression_ratio': compression_ratio,
        'composite_count': composite_count,
        'composite_ratio': composite_ratio
    }


# Example usage
if __name__ == '__main__':
    # Sample surgical action sequences
    sequences = [
        ['pick', 'grasp', 'pull', 'cut', 'drop'],
        ['pick', 'grasp', 'pull', 'cut', 'drop'],
        ['pick', 'grasp', 'push', 'cut', 'drop'],
        ['grasp', 'pull', 'cut'],
    ]
    
    # Mine n-grams
    miner = NGramMiner(min_freq=2, max_ngram_size=3)
    all_ngrams = miner.mine_all_ngrams(sequences)
    
    # Print statistics
    stats = miner.get_statistics(all_ngrams)
    print("N-gram mining statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Create composite symbols
    pattern_to_composite = {}
    composite_id = 0
    for n, ngrams in all_ngrams.items():
        for pattern in ngrams.keys():
            pattern_to_composite[pattern] = f'C{composite_id}'
            composite_id += 1
    
    # Replace patterns
    matcher = GreedyMatcher(pattern_to_composite)
    replaced = matcher.replace_in_sequences(sequences)
    
    print("\nReplacement example:")
    print(f"Original: {sequences[0]}")
    print(f"Replaced: {replaced[0]}")

"""
Main grammar induction algorithm for PTG.

Combines all components (n-gram mining, transition model, duration model,
chirality lexicon, object model, goal matrices) into unified grammar.

Paper: Algorithm 1 - Goal-Conditioned Probabilistic Temporal Grammar Induction
"""
from typing import List, Dict, Tuple, Optional
import pickle
from pathlib import Path
from dataclasses import dataclass
import numpy as np

from .ngram_miner import NGramMiner, GreedyMatcher
from .transition_model import TransitionModel
from .duration_model import DurationModel
from .chirality_lexicon import ChiralityLexicon
from .grammar_scorer import GrammarScorer


@dataclass
class ActionSegment:
    """Represents an annotated action segment."""
    action: str
    start_frame: int
    end_frame: int
    objects: Optional[List[str]] = None
    duration: Optional[int] = None
    
    def __post_init__(self):
        if self.duration is None:
            self.duration = self.end_frame - self.start_frame + 1


class ProbabilisticTemporalGrammar:
    """
    Complete PTG grammar structure.
    
    Paper: "Grammar G = {P, D, C, O, γ, G, R}"
    """
    
    def __init__(
        self,
        transition_model: TransitionModel,
        duration_model: DurationModel,
        chirality_lexicon: ChiralityLexicon,
        composite_symbols: Dict[Tuple[str, ...], str],
        object_model: Optional[Dict] = None,
        goal_model: Optional[Dict] = None,
        scorer: Optional[GrammarScorer] = None
    ):
        self.P = transition_model  # Transition probabilities
        self.D = duration_model    # Duration statistics
        self.C = composite_symbols # Composite symbols
        self.gamma = chirality_lexicon  # Chirality pairs
        self.O = object_model or {}     # Object transitions
        self.G = goal_model or {}       # Goal matrices
        self.scorer = scorer
    
    def save(self, filepath: str):
        """Save complete grammar to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'transition_model': self.P,
                'duration_model': self.D,
                'composite_symbols': self.C,
                'chirality_lexicon': self.gamma,
                'object_model': self.O,
                'goal_model': self.G
            }, f)
        
        print(f"Grammar saved to {filepath}")
        print(f"  Vocabulary size: {len(self.P.vocabulary)}")
        print(f"  Composite symbols: {len(self.C)}")
        print(f"  Chirality pairs: {len(self.gamma.pairs)}")
    
    @classmethod
    def load(cls, filepath: str) -> 'ProbabilisticTemporalGrammar':
        """Load grammar from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        return cls(
            transition_model=data['transition_model'],
            duration_model=data['duration_model'],
            composite_symbols=data['composite_symbols'],
            chirality_lexicon=data['chirality_lexicon'],
            object_model=data.get('object_model'),
            goal_model=data.get('goal_model')
        )


class GrammarInducer:
    """
    Main grammar induction algorithm.
    
    Paper: "The induction algorithm proceeds bottom-up, iteratively mine 
    frequent contiguous subsequences (n-grams), abstract them into unique 
    composite tokens (stored in C), and replace occurrences in the corpus"
    """
    
    def __init__(
        self,
        min_ngram_freq: int = 3,
        max_ngram_size: int = 4,
        markov_order: int = 2,
        chirality_prior_boost: float = 1.2
    ):
        self.min_ngram_freq = min_ngram_freq
        self.max_ngram_size = max_ngram_size
        self.markov_order = markov_order
        self.chirality_prior_boost = chirality_prior_boost
    
    def induce(
        self,
        action_segments: List[ActionSegment]
    ) -> ProbabilisticTemporalGrammar:
        """
        Induce grammar from annotated corpus.
        
        Args:
            action_segments: List of action segments with annotations
            
        Returns:
            Complete PTG grammar
        """
        print("=" * 60)
        print("GRAMMAR INDUCTION")
        print("=" * 60)
        
        # Step 1: Extract sequences
        sequences = [[seg.action for seg in action_segments]]
        print(f"\n1. Parsed {len(sequences)} sequences")
        print(f"   Total actions: {sum(len(seq) for seq in sequences)}")
        
        # Step 2: Mine n-grams
        print(f"\n2. Mining n-grams (freq >= {self.min_ngram_freq})...")
        miner = NGramMiner(self.min_ngram_freq, self.max_ngram_size)
        all_ngrams = miner.mine_all_ngrams(sequences)
        
        # Step 3: Create composite symbols
        print("\n3. Creating composite symbols...")
        composite_symbols = {}
        composite_id = 0
        
        for n in sorted(all_ngrams.keys()):
            for pattern in all_ngrams[n].keys():
                composite_symbols[pattern] = f"C{composite_id}"
                composite_id += 1
        
        print(f"   Created {len(composite_symbols)} composite symbols")
        
        # Step 4: Replace patterns in sequences
        print("\n4. Replacing patterns with composite symbols...")
        matcher = GreedyMatcher(composite_symbols)
        abstracted_sequences = matcher.replace_in_sequences(sequences)
        
        # Step 5: Initialize chirality lexicon
        print("\n5. Initializing chirality lexicon...")
        chirality_lexicon = ChiralityLexicon(self.chirality_prior_boost)
        print(f"   Loaded {len(chirality_lexicon.pairs)} chiral pairs")
        
        # Step 6: Estimate transition probabilities
        print(f"\n6. Estimating {self.markov_order}-order transition probabilities...")
        transition_model = TransitionModel(markov_order=self.markov_order)
        transition_model.fit(abstracted_sequences)
        stats = transition_model.get_statistics()
        print(f"   Vocabulary: {stats['vocabulary_size']} actions")
        print(f"   Histories: {stats['num_histories']}")
        
        # Step 7: Collect duration statistics
        print("\n7. Computing duration statistics...")
        duration_model = DurationModel()
        segment_tuples = [
            (seg.action, seg.start_frame, seg.end_frame)
            for seg in action_segments
        ]
        duration_model.fit(segment_tuples)
        dur_summary = duration_model.get_summary()
        print(f"   Actions with duration stats: {dur_summary['num_actions']}")
        print(f"   Mean duration: {dur_summary['median_duration_mean']:.1f} frames")
        
        # Step 8: Object transitions (placeholder)
        print("\n8. Estimating object transitions...")
        object_model = self._estimate_object_model(action_segments)
        print(f"   Object states tracked: {len(object_model) if object_model else 0}")
        
        # Step 9: Goal matrices (placeholder)
        print("\n9. Constructing goal matrices...")
        goal_model = self._construct_goal_matrices(action_segments)
        print(f"   Goal states: {len(goal_model) if goal_model else 0}")
        
        # Step 10: Create grammar
        print("\n10. Assembling complete grammar...")
        grammar = ProbabilisticTemporalGrammar(
            transition_model=transition_model,
            duration_model=duration_model,
            chirality_lexicon=chirality_lexicon,
            composite_symbols=composite_symbols,
            object_model=object_model,
            goal_model=goal_model
        )
        
        print("\n" + "=" * 60)
        print("GRAMMAR INDUCTION COMPLETE")
        print("=" * 60)
        
        return grammar
    
    def _estimate_object_model(
        self,
        action_segments: List[ActionSegment]
    ) -> Dict:
        """
        Estimate object transition statistics.
        
        Paper: "Estimate obj-conditioned transitions P(o_t | o_{t-1}, a_t) to form O"
        """
        # Placeholder: In full implementation, track object state changes
        object_model = {}
        
        for seg in action_segments:
            if seg.objects:
                # Create binary vector for objects
                object_model[seg.action] = np.zeros(16)  # Placeholder
        
        return object_model
    
    def _construct_goal_matrices(
        self,
        action_segments: List[ActionSegment]
    ) -> Dict:
        """
        Construct goal matrices G, R.
        
        Paper: "Construct goal matrices G^{jk} from object-action co-occurrence.
        Learn soft goal reward matrices R^{jk} (e.g., cross-entropy parameterization)"
        """
        # Placeholder: In full implementation, use GoMMC algorithm
        goal_model = {
            'G': {},  # Binary reachability
            'R': {}   # Learned rewards
        }
        
        return goal_model


# Example usage
if __name__ == '__main__':
    # Sample data
    segments = [
        ActionSegment('pick', 0, 10),
        ActionSegment('grasp', 11, 50),
        ActionSegment('pull', 51, 70),
        ActionSegment('cut', 71, 90),
        ActionSegment('drop', 91, 100),
        ActionSegment('pick', 101, 112),
        ActionSegment('grasp', 113, 145),
        ActionSegment('push', 146, 168),
        ActionSegment('cut', 169, 190),
        ActionSegment('drop', 191, 200),
    ]
    
    # Induce grammar
    inducer = GrammarInducer(
        min_ngram_freq=2,
        max_ngram_size=3,
        markov_order=2,
        chirality_prior_boost=1.2
    )
    
    grammar = inducer.induce(segments)
    
    # Save
    grammar.save('test_grammar.pkl')
    
    # Load
    loaded_grammar = ProbabilisticTemporalGrammar.load('test_grammar.pkl')
    print("\nGrammar loaded successfully!")

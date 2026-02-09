"""
Chirality lexicon for surgical actions.

Defines chiral action pairs (temporal reversals) that are semantically
distinct but visually subtle, such as push_needle <-> pull_suture.

Paper: "Chirality pairs encode locally important temporal reversals that 
may be semantically crucial but infrequent."
"""
from typing import Dict, Set, Tuple, List
from dataclasses import dataclass


@dataclass
class ChiralPair:
    """Represents a chiral action pair."""
    action1: str
    action2: str
    prior_boost: float = 1.0  # Multiplicative boost for coverage


class ChiralityLexicon:
    """
    Domain lexicon for chiral surgical action pairs.
    
    Paper: "matching verbs to a small domain lexicon (e.g., pick<->drop, 
    push<->pull) and assigning a small prior boost to those inverses to 
    guarantee coverage even when rare"
    """
    
    def __init__(self, prior_boost: float = 1.2):
        """
        Args:
            prior_boost: Prior probability boost for chiral pairs
        """
        self.prior_boost = prior_boost
        self.pairs: List[ChiralPair] = []
        self._pair_map: Dict[str, str] = {}
        
        # Initialize with common surgical chiral pairs
        self._initialize_surgical_pairs()
    
    def _initialize_surgical_pairs(self):
        """Initialize common surgical chiral action pairs."""
        # Verb-level chirality
        verb_pairs = [
            ('pick', 'drop'),
            ('push', 'pull'),
            ('insert', 'retract'),
            ('open', 'close'),
            ('cut', 'suture'),
            ('grasp', 'release'),
            ('advance', 'withdraw'),
            ('in', 'out'),
            ('up', 'down'),
            ('left', 'right'),
        ]
        
        for action1, action2 in verb_pairs:
            self.add_pair(action1, action2)
    
    def add_pair(
        self,
        action1: str,
        action2: str,
        boost: float = None
    ):
        """
        Add a chiral action pair.
        
        Args:
            action1: First action in pair
            action2: Second action (chiral inverse)
            boost: Optional custom prior boost
        """
        boost = boost if boost is not None else self.prior_boost
        
        pair = ChiralPair(action1, action2, boost)
        self.pairs.append(pair)
        
        # Bidirectional mapping
        self._pair_map[action1] = action2
        self._pair_map[action2] = action1
    
    def get_inverse(self, action: str) -> str:
        """
        Get the chiral inverse of an action.
        
        Args:
            action: Action to find inverse for
            
        Returns:
            Chiral inverse action, or None if not in lexicon
        """
        return self._pair_map.get(action)
    
    def is_chiral_pair(self, action1: str, action2: str) -> bool:
        """Check if two actions form a chiral pair."""
        return (action1 in self._pair_map and 
                self._pair_map[action1] == action2)
    
    def get_chirality_prior(
        self,
        prev_action: str,
        curr_action: str
    ) -> float:
        """
        Get prior probability boost for a transition.
        
        Paper: γ(a_t-1, a) provides chirality prior
        
        Args:
            prev_action: Previous action
            curr_action: Current action
            
        Returns:
            Prior boost (>1.0 if chiral pair, 1.0 otherwise)
        """
        if self.is_chiral_pair(prev_action, curr_action):
            # Find the specific pair
            for pair in self.pairs:
                if ((pair.action1 == prev_action and pair.action2 == curr_action) or
                    (pair.action2 == prev_action and pair.action1 == curr_action)):
                    return pair.prior_boost
        
        return 1.0
    
    def get_all_pairs(self) -> List[Tuple[str, str]]:
        """Get all chiral pairs."""
        return [(p.action1, p.action2) for p in self.pairs]
    
    def expand_with_triplets(
        self,
        instruments: List[str],
        targets: List[str]
    ):
        """
        Expand chirality lexicon to handle triplet format.
        
        For datasets like CholecT50 with (verb, instrument, target) triplets,
        we expand verb-level chirality to full triplets.
        
        Args:
            instruments: List of instrument names
            targets: List of target anatomies
        """
        base_pairs = list(self.pairs)  # Copy current pairs
        
        for pair in base_pairs:
            verb1, verb2 = pair.action1, pair.action2
            
            # Create triplet pairs for all combinations
            for instrument in instruments:
                for target in targets:
                    triplet1 = f"{verb1}_{instrument}_{target}"
                    triplet2 = f"{verb2}_{instrument}_{target}"
                    self.add_pair(triplet1, triplet2, pair.prior_boost)
    
    def to_dict(self) -> Dict:
        """Export lexicon to dictionary."""
        return {
            'pairs': [(p.action1, p.action2, p.prior_boost) for p in self.pairs],
            'prior_boost': self.prior_boost
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ChiralityLexicon':
        """Load lexicon from dictionary."""
        lexicon = cls(prior_boost=data['prior_boost'])
        lexicon.pairs = []  # Clear default pairs
        lexicon._pair_map = {}
        
        for action1, action2, boost in data['pairs']:
            lexicon.add_pair(action1, action2, boost)
        
        return lexicon


# Example usage
if __name__ == '__main__':
    # Create lexicon
    lexicon = ChiralityLexicon(prior_boost=1.2)
    
    # Check some pairs
    print("Chiral pairs:")
    for action1, action2 in lexicon.get_all_pairs():
        print(f"  {action1} <-> {action2}")
    
    # Test chirality detection
    print("\nChirality tests:")
    print(f"push -> pull: {lexicon.is_chiral_pair('push', 'pull')}")
    print(f"push -> cut: {lexicon.is_chiral_pair('push', 'cut')}")
    
    # Test prior boost
    print("\nPrior boosts:")
    print(f"pick -> drop: {lexicon.get_chirality_prior('pick', 'drop')}")
    print(f"pick -> cut: {lexicon.get_chirality_prior('pick', 'cut')}")
    
    # Expand to triplets
    instruments = ['grasper', 'scissors']
    targets = ['gallbladder', 'cystic_duct']
    lexicon.expand_with_triplets(instruments, targets)
    
    print(f"\nTotal pairs after triplet expansion: {len(lexicon.pairs)}")

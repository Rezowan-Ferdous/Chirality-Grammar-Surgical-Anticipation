"""
Probabilistic Earley Parser for PTG.

Implements polynomial-time parsing for near-regular grammars with
probabilistic scoring for procedurally valid action sequences.

Paper: "At inference, a probabilistic Earley-style parser filters
candidate sequences by procedural validity, scoring each by
Σ[log S_N(a_τ) + s_G(a_τ|h_τ)]"

Reference: KARI parser (https://github.com/gongda0e/KARI)
"""
import torch
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class EarleyState:
    """Represents a state in the Earley chart."""
    rule_lhs: str  # Left-hand side (non-terminal)
    rule_rhs: Tuple[str, ...]  # Right-hand side (sequence of symbols)
    dot_position: int  # Position of the dot in the rule
    start_position: int  # Starting position in input
    end_position: int  # Ending position in input
    score: float = 0.0  # Log probability score
    
    def __hash__(self):
        return hash((self.rule_lhs, self.rule_rhs, self.dot_position, 
                    self.start_position, self.end_position))
    
    def __eq__(self, other):
        return (self.rule_lhs == other.rule_lhs and
                self.rule_rhs == other.rule_rhs and
                self.dot_position == other.dot_position and
                self.start_position == other.start_position and
                self.end_position == other.end_position)
    
    def is_complete(self) -> bool:
        """Check if dot is at the end (rule is complete)."""
        return self.dot_position >= len(self.rule_rhs)
    
    def next_symbol(self) -> Optional[str]:
        """Get symbol after the dot."""
        if self.is_complete():
            return None
        return self.rule_rhs[self.dot_position]


class ProbabilisticEarleyParser:
    """
    Probabilistic Earley parser for PTG.
    
    Paper: "The Earley parser runs in polynomial time O(|π|²) for
    near-regular grammars and enforces procedural validity"
    """
    
    def __init__(
        self,
        grammar_rules: Dict[str, List[Tuple[str, ...]]],
        transition_probs: Dict[Tuple[str, ...], Dict[str, float]],
        start_symbol: str = 'S'
    ):
        """
        Args:
            grammar_rules: Dict mapping non-terminals to production rules
            transition_probs: Transition probabilities from grammar
            start_symbol: Start symbol for parsing
        """
        self.grammar_rules = grammar_rules
        self.transition_probs = transition_probs
        self.start_symbol = start_symbol
        
        # Determine terminals vs non-terminals
        self.non_terminals = set(grammar_rules.keys())
        self.terminals = set()
        for rules in grammar_rules.values():
            for rhs in rules:
                for symbol in rhs:
                    if symbol not in self.non_terminals:
                        self.terminals.add(symbol)
    
    def parse(
        self,
        input_sequence: List[str],
        neural_scores: Optional[torch.Tensor] = None,
        grammar_scorer = None
    ) -> List[EarleyState]:
        """
        Parse an input sequence with Earley algorithm.
        
        Args:
            input_sequence: Sequence of action symbols
            neural_scores: (T, num_actions) neural network scores
            grammar_scorer: GrammarScorer for computing s_G
            
        Returns:
            List of completed parse states
        """
        n = len(input_sequence)
        
        # Initialize chart: chart[i] contains states ending at position i
        chart = [set() for _ in range(n + 1)]
        
        # Add initial state: S → • α
        for rhs in self.grammar_rules.get(self.start_symbol, []):
            initial_state = EarleyState(
                rule_lhs=self.start_symbol,
                rule_rhs=rhs,
                dot_position=0,
                start_position=0,
                end_position=0,
                score=0.0
            )
            chart[0].add(initial_state)
        
        # Process each position
        for i in range(n + 1):
            # Process all states at position i (can grow during iteration)
            states_to_process = list(chart[i])
            
            for state in states_to_process:
                if state.is_complete():
                    # Completer: state is A → α •
                    self._completer(state, chart)
                else:
                    next_sym = state.next_symbol()
                    
                    if next_sym in self.non_terminals:
                        # Predictor: state is A → α • B β
                        self._predictor(state, chart, i)
                    elif i < n and next_sym == input_sequence[i]:
                        # Scanner: state is A → α • a β, and next input is a
                        self._scanner(state, chart, i, input_sequence[i],
                                    neural_scores, grammar_scorer)
        
        # Return completed parses spanning entire input
        completed = [s for s in chart[n] 
                    if s.is_complete() and 
                    s.rule_lhs == self.start_symbol and
                    s.start_position == 0]
        
        return completed
    
    def _predictor(
        self,
        state: EarleyState,
        chart: List[Set[EarleyState]],
        position: int
    ):
        """
        Predictor: For state A → α • B β, add B → • γ for all rules B → γ.
        """
        next_sym = state.next_symbol()
        
        for rhs in self.grammar_rules.get(next_sym, []):
            new_state = EarleyState(
                rule_lhs=next_sym,
                rule_rhs=rhs,
                dot_position=0,
                start_position=position,
                end_position=position,
                score=0.0  # Initialize score
            )
            
            if new_state not in chart[position]:
                chart[position].add(new_state)
    
    def _scanner(
        self,
        state: EarleyState,
        chart: List[Set[EarleyState]],
        position: int,
        input_symbol: str,
        neural_scores: Optional[torch.Tensor],
        grammar_scorer
    ):
        """
        Scanner: For state A → α • a β, advance dot if next input is a.
        
        Compute score: log S_N(a) + s_G(a | history)
        """
        # Compute score for this action
        action_score = 0.0
        
        # Neural score
        if neural_scores is not None:
            # Assuming neural_scores[position] is logits for actions
            action_idx = self._action_to_idx(input_symbol)
            if action_idx is not None:
                action_score += neural_scores[position, action_idx].item()
        
        # Grammar score
        if grammar_scorer is not None:
            # Get history from input up to position
            history = tuple(chart[0][0].rule_rhs[:position]) if position > 0 else tuple()
            grammar_score = grammar_scorer.score_action(input_symbol, history)
            action_score += grammar_score
        
        # Create new state with dot advanced
        new_state = EarleyState(
            rule_lhs=state.rule_lhs,
            rule_rhs=state.rule_rhs,
            dot_position=state.dot_position + 1,
            start_position=state.start_position,
            end_position=position + 1,
            score=state.score + action_score
        )
        
        chart[position + 1].add(new_state)
    
    def _completer(
        self,
        state: EarleyState,
        chart: List[Set[EarleyState]]
    ):
        """
        Completer: For completed state B → γ •, advance all states
        A → α • B β in chart[state.start_position].
        """
        # Find all states waiting for this non-terminal
        for waiting_state in list(chart[state.start_position]):
            if (not waiting_state.is_complete() and
                waiting_state.next_symbol() == state.rule_lhs):
                
                # Advance dot in waiting state
                new_state = EarleyState(
                    rule_lhs=waiting_state.rule_lhs,
                    rule_rhs=waiting_state.rule_rhs,
                    dot_position=waiting_state.dot_position + 1,
                    start_position=waiting_state.start_position,
                    end_position=state.end_position,
                    score=waiting_state.score + state.score
                )
                
                chart[state.end_position].add(new_state)
    
    def _action_to_idx(self, action: str) -> Optional[int]:
        """Map action string to index (placeholder)."""
        # Would need actual vocabulary mapping
        return hash(action) % 466  # Dummy mapping
    
    def beam_search(
        self,
        initial_obs: List[str],
        max_length: int,
        beam_size: int,
        neural_model,
        grammar_scorer
    ) -> List[Tuple[List[str], float]]:
        """
        Beam search with grammar constraints.
        
        Args:
            initial_obs: Observed action sequence
            max_length: Maximum prediction length
            beam_size: Beam width
            neural_model: Neural model for next action prediction
            grammar_scorer: Grammar scorer
            
        Returns:
            List of (sequence, score) tuples
        """
        # Initialize beam with empty sequence
        beam = [(initial_obs, 0.0)]
        
        for step in range(max_length):
            candidates = []
            
            for sequence, score in beam:
                # Get neural predictions
                with torch.no_grad():
                    # Placeholder: would call neural model
                    neural_probs = torch.softmax(torch.randn(466), dim=0)
                
                # Get grammar distribution
                history = tuple(sequence[-2:]) if len(sequence) >= 2 else tuple(sequence)
                grammar_dist = grammar_scorer.get_distribution(history)
                
                # Combine scores for top-k actions
                top_k = 10
                top_actions = sorted(grammar_dist.keys(), 
                                   key=lambda a: grammar_dist[a], 
                                   reverse=True)[:top_k]
                
                for action in top_actions:
                    # Combined score
                    neural_score = torch.log(neural_probs[hash(action) % 466])
                    grammar_score = np.log(grammar_dist[action] + 1e-10)
                    combined_score = neural_score.item() + grammar_score
                    
                    # Check procedural validity with parser
                    candidate_seq = sequence + [action]
                    if self._is_valid_sequence(candidate_seq):
                        candidates.append((candidate_seq, score + combined_score))
            
            # Select top beam_size candidates
            beam = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
        
        return beam
    
    def _is_valid_sequence(self, sequence: List[str]) -> bool:
        """Check if sequence is procedurally valid."""
        # Simplified: check if parseable
        parses = self.parse(sequence)
        return len(parses) > 0


# Example usage
if __name__ == '__main__':
    # Define a simple grammar
    # S → A B
    # A → pick | grasp
    # B → cut | drop
    grammar_rules = {
        'S': [('A', 'B')],
        'A': [('pick',), ('grasp',)],
        'B': [('cut',), ('drop',)]
    }
    
    # Dummy transition probabilities
    transition_probs = {
        tuple(): {'pick': 0.5, 'grasp': 0.5},
        ('pick',): {'cut': 0.7, 'drop': 0.3},
        ('grasp',): {'cut': 0.6, 'drop': 0.4}
    }
    
    # Create parser
    parser = ProbabilisticEarleyParser(
        grammar_rules=grammar_rules,
        transition_probs=transition_probs,
        start_symbol='S'
    )
    
    # Test parsing
    valid_sequence = ['pick', 'cut']
    invalid_sequence = ['pick', 'pick']
    
    print("Testing Earley parser:")
    print(f"\nValid sequence: {valid_sequence}")
    parses = parser.parse(valid_sequence)
    print(f"  Parses found: {len(parses)}")
    
    print(f"\nInvalid sequence: {invalid_sequence}")
    parses = parser.parse(invalid_sequence)
    print(f"  Parses found: {len(parses)}")

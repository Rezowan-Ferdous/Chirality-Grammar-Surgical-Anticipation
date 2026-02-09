"""
Grammar induction script.

Induces probabilistic temporal grammar from surgical action annotations.

Usage:
    python scripts/induce_grammar.py \
        --corpus_path data/cholec80_annotations \
        --output_path grammars/cholec80_ptg.pkl
"""
import argparse
from pathlib import Path
import json
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from grammar.grammar_inducer import GrammarInducer, ActionSegment, ProbabilisticTemporalGrammar


def load_annotations(corpus_path: Path):
    """Load surgical action annotations."""
    annotations = []
    
    # Try to load JSON annotations
    json_files = list(corpus_path.glob('*.json'))
    
    if json_files:
        for json_file in json_files:
            with open(json_file) as f:
                data = json.load(f)
                annotations.extend(data)
    else:
        print(f"Warning: No JSON files found in {corpus_path}")
    
    return annotations


def convert_to_segments(annotations):
    """Convert annotations to ActionSegment objects."""
    segments = []
    
    for anno in annotations:
        # Handle different annotation formats
        if 'segments' in anno:
            for seg in anno['segments']:
                segments.append(ActionSegment(
                    action=seg['action'],
                    start_frame=seg['start_frame'],
                    end_frame=seg['end_frame'],
                    objects=seg.get('objects'),
                    duration=seg.get('duration')
                ))
        else:
            # Simple format: just action and frames
            segments.append(ActionSegment(
                action=anno.get('action', 'unknown'),
                start_frame=anno.get('start_frame', 0),
                end_frame=anno.get('end_frame', 100)
            ))
    
    return segments


def main():
    parser = argparse.ArgumentParser(description='Induce PTG grammar from corpus')
    parser.add_argument('--corpus_path', type=str, required=True,
                       help='Path to annotation directory')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Output path for grammar (.pkl)')
    parser.add_argument('--min_ngram_freq', type=int, default=3,
                       help='Minimum n-gram frequency')
    parser.add_argument('--max_ngram_size', type=int, default=4,
                       help='Maximum n-gram size')
    parser.add_argument('--markov_order', type=int, default=2,
                       help='Markov model order')
    parser.add_argument('--chirality_boost', type=float, default=1.2,
                       help='Chirality prior boost')
    args = parser.parse_args()
    
    corpus_path = Path(args.corpus_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading annotations from {corpus_path}...")
    annotations = load_annotations(corpus_path)
    
    if not annotations:
        print("No annotations found! Using dummy data for demonstration.")
        # Create dummy segments
        segments = [
            ActionSegment('pick', 0, 10),
            ActionSegment('grasp', 11, 50),
            ActionSegment('pull', 51, 70),
            ActionSegment('cut', 71, 90),
            ActionSegment('drop', 91, 100),
        ] * 5  # Repeat for frequency
    else:
        print(f"Found {len(annotations)} annotations")
        segments = convert_to_segments(annotations)
    
    print(f"Total action segments: {len(segments)}")
    
    # Create inducer
    inducer = GrammarInducer(
        min_ngram_freq=args.min_ngram_freq,
        max_ngram_size=args.max_ngram_size,
        markov_order=args.markov_order,
        chirality_prior_boost=args.chirality_boost
    )
    
    # Induce grammar
    print("\nInducing grammar...")
    grammar = inducer.induce(segments)
    
    # Save
    grammar.save(str(output_path))
    
    print(f"\n✓ Grammar saved to {output_path}")
    print(f"  Vocabulary: {len(grammar.P.vocabulary)} actions")
    print(f"  Composite symbols: {len(grammar.C)}")
    print(f"  Chirality pairs: {len(grammar.gamma.pairs)}")


if __name__ == '__main__':
    main()

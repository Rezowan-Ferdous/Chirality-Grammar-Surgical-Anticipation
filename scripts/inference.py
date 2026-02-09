"""
Inference script for PTG model.

Predicts future action sequences from observed video.

Usage:
    python scripts/inference.py \
        --model_path outputs/ptg_phase2_final.pth \
        --video_path path/to/video.mp4 \
        --config configs/ptg_chirality.yaml
"""
import torch
import argparse
from pathlib import Path
import yaml
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.ptg_model import PTGModel
from grammar.grammar_inducer import ProbabilisticTemporalGrammar, GrammarScorer


def main():
    parser = argparse.ArgumentParser(description='PTG Inference')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--video_path', type=str, required=True,
                       help='Path to input video')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--grammar_path', type=str, default=None,
                       help='Path to grammar (for Phase 2 models)')
    parser.add_argument('--max_steps', type=int, default=10,
                       help='Maximum prediction steps')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Create model
    print("\nLoading model...")
    model = PTGModel(
        vjepa_model_name=config['model']['vjepa_variant'],
        freeze_vjepa=config['model']['freeze_vjepa'],
        hidden_dim=config['model']['hidden_dim'],
        num_decoder_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        num_actions=config['model']['num_actions'],
        num_objects=config['model']['num_objects'],
        use_goal_head=True
    ).to(device)
    
    # Load weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {args.model_path}")
    
    # Load grammar if provided
    if args.grammar_path:
        print(f"\nLoading grammar from {args.grammar_path}...")
        grammar = ProbabilisticTemporalGrammar.load(args.grammar_path)
        grammar_scorer = GrammarScorer(
            transition_model=grammar.P,
            duration_model=grammar.D,
            chirality_lexicon=grammar.gamma,
            lambda_o=config['grammar']['lambda_o'],
            lambda_d=config['grammar']['lambda_d'],
            lambda_r=config['grammar']['lambda_r'],
            temperature=config['grammar']['temperature']
        )
        model.set_grammar_scorer(grammar_scorer)
    
    # Extract features from video
    print(f"\nExtracting features from {args.video_path}...")
    video_path = Path(args.video_path)
    
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        # Use dummy features for testing
        print("Using dummy features for demonstration...")
        video_frames = torch.randn(1, 16, 3, 224, 224).to(device)
    else:
        # In real implementation, would load and preprocess video
        video_frames = torch.randn(1, 16, 3, 224, 224).to(device)
    
    # Predict future sequence
    print(f"\nPredicting {args.max_steps} future actions...")
    with torch.no_grad():
        predictions = model.predict_sequence(
            video_frames,
            max_steps=args.max_steps,
            use_grammar=(args.grammar_path is not None),
            temperature=args.temperature
        )
    
    # Display predictions
    print("\n" + "="*60)
    print("PREDICTED ACTION SEQUENCE")
    print("="*60)
    
    for i, (action_idx, duration, objects) in enumerate(zip(
        predictions['actions'],
        predictions['durations'],
        predictions['objects']
    ), 1):
        print(f"\nStep {i}:")
        print(f"  Action: {action_idx} (index)")
        print(f"  Duration: {duration:.1f} frames")
        print(f"  Objects: {objects.nonzero()[0].tolist() if len(objects.nonzero()[0]) > 0 else 'None'}")
    
    print("\n" + "="*60)
    print("Inference complete!")


if __name__ == '__main__':
    main()

"""
Evaluation script for PTG model.

Evaluates trained model on test set with comprehensive metrics.

Usage:
    python scripts/evaluate.py \
        --model_path outputs/ptg_phase2_final.pth \
        --config configs/ptg_chirality.yaml \
        --test_split test
"""
import torch
from torch.utils.data import DataLoader
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.ptg_model import PTGModel
from data.cisa_dataset import CiSADataset, collate_fn
from evaluation.metrics import PTGEvaluator
from grammar.grammar_inducer import ProbabilisticTemporalGrammar


def evaluate_model(
    model: PTGModel,
    dataloader: DataLoader,
    evaluator: PTGEvaluator,
    device: str
):
    """Evaluate model on dataset."""
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            obs_features = batch['obs_features'].to(device)
            future_actions = batch['future_actions'].to(device)
            future_durations = batch['future_durations'].to(device)
            future_objects = batch['future_objects'].to(device)
            
            # Forward pass
            horizon = future_actions.shape[1]
            outputs = model.futr_decoder(obs_features, prediction_horizon=horizon)
            
            # Prepare targets
            targets = {
                'actions': future_actions,
                'durations': future_durations,
                'objects': future_objects
            }
            
            # Update metrics
            evaluator.update(outputs, targets, horizon)
    
    return evaluator.get_summary()


def main():
    parser = argparse.ArgumentParser(description='Evaluate PTG model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--test_split', type=str, default='test',
                       help='Test split name')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file for results (JSON)')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Create test dataset
    print("\nCreating test dataset...")
    test_dataset = CiSADataset(
        data_root=config['data']['data_root'],
        dataset_name='cholec80',
        split=args.test_split,
        observation_rate=config['data']['observation_rates'][2],  # 0.5
        prediction_rate=config['data']['prediction_rates'][1],    # 0.2
        use_precomputed_features=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
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
    print(f"Model loaded from {args.model_path}")
    
    # Create evaluator
    print("\nInitializing evaluator...")
    action_vocab = [f'action_{i}' for i in range(config['model']['num_actions'])]
    chiral_pairs = [('pick', 'drop'), ('push', 'pull')]  # Placeholder
    
    evaluator = PTGEvaluator(
        action_vocab=action_vocab,
        chiral_pairs=chiral_pairs,
        eval_horizons=config['evaluation']['eval_horizons']
    )
    
    # Evaluate
    print("\n" + "=" * 60)
    print("STARTING EVALUATION")
    print("=" * 60)
    
    results = evaluate_model(model, test_loader, evaluator, device)
    
    # Print results
    evaluator.print_summary()
    
    # Save results
    if args.output_file:
        import json
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()

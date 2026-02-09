"""
Training script for PTG model.

Implements 2-phase training:
- Phase 1: Supervised pre-training (30 epochs)
- Phase 2: Grammar-regularized training (30 epochs)

Usage:
    python scripts/train_ptg.py --config configs/ptg_chirality.yaml
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.ptg_model import PTGModel
from training.losses import PTGLoss
from data.cisa_dataset import CiSADataset, collate_fn
from grammar.grammar_inducer import ProbabilisticTemporalGrammar, GrammarScorer


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: PTGLoss,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    grammar_scorer: GrammarScorer = None
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    loss_components = {}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        # Move to device
        obs_features = batch['obs_features'].to(device)
        future_actions = batch['future_actions'].to(device)
        future_durations = batch['future_durations'].to(device)
        future_objects = batch['future_objects'].to(device)
        fut_mask = batch['fut_mask'].to(device)
        
        # Forward pass
        outputs = model.futr_decoder(obs_features, prediction_horizon=future_actions.shape[1])
        
        # Prepare targets
        targets = {
            'actions': future_actions,
            'durations': future_durations,
            'objects': future_objects,
            'duration_mask': ~fut_mask
        }
        
        # Get grammar probs if Phase 2
        grammar_probs = None
        if grammar_scorer is not None and criterion.use_grammar:
            # Placeholder: would need action history
            pass
        
        # Compute loss
        losses = criterion(outputs, targets, grammar_probs)
        
        # Backward
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()
        
        # Track losses
        total_loss += losses['total'].item()
        for key, value in losses.items():
            if key not in loss_components:
                loss_components[key] = 0
            loss_components[key] += value.item()
        
        pbar.set_postfix({'loss': losses['total'].item()})
    
    # Average losses
    n_batches = len(dataloader)
    avg_loss = total_loss / n_batches
    avg_components = {k: v / n_batches for k, v in loss_components.items()}
    
    return {'total': avg_loss, **avg_components}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--grammar_path', type=str, default=None, help='Path to pre-induced grammar')
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2], help='Training phase')
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = CiSADataset(
        data_root=config['data']['data_root'],
        dataset_name='cholec80',
        split='train',
        observation_rate=config['data']['observation_rates'][2],  # 0.5
        prediction_rate=config['data']['prediction_rates'][1],    # 0.2
        use_precomputed_features=False
    )
    
    val_dataset = CiSADataset(
        data_root=config['data']['data_root'],
        dataset_name='cholec80',
        split='val',
        observation_rate=config['data']['observation_rates'][2],
        prediction_rate=config['data']['prediction_rates'][1],
        use_precomputed_features=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        collate_fn=collate_fn
    )
    
    # Create model
    print("\nCreating PTG model...")
    model = PTGModel(
        vjepa_model_name=config['model']['vjepa_variant'],
        freeze_vjepa=config['model']['freeze_vjepa'],
        hidden_dim=config['model']['hidden_dim'],
        num_decoder_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        dim_feedforward=config['model']['dim_feedforward'],
        dropout=config['model']['dropout'],
        num_actions=config['model']['num_actions'],
        num_objects=config['model']['num_objects'],
        num_goals=config['model']['num_goals'],
        use_goal_head=True
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load grammar for Phase 2
    grammar_scorer = None
    if args.phase == 2 and args.grammar_path:
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
    
    # Create loss function
    criterion = PTGLoss(
        lambda_dur=config['training']['lambda_dur'],
        lambda_obj=config['training']['lambda_obj'],
        lambda_goal=config['training']['lambda_goal'],
        lambda_gram=config['training'].get('lambda_gram', 0.5),
        use_grammar=(args.phase == 2),
        use_goal=True
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Training loop
    num_epochs = config['training']['pretrain_epochs'] if args.phase == 1 else config['training']['num_epochs']
    
    print(f"\n{'='*60}")
    print(f"PHASE {args.phase} TRAINING")
    print(f"{'='*60}\n")
    
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch, grammar_scorer
        )
        
        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"  Train loss: {train_metrics['total']:.4f}")
        for key, value in train_metrics.items():
            if key != 'total':
                print(f"    {key}: {value:.4f}")
        
        # Save checkpoint
        if epoch % config['training']['save_interval'] == 0:
            checkpoint_path = output_dir / f"checkpoint_phase{args.phase}_epoch{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics['total']
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_path = output_dir / f"ptg_phase{args.phase}_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")


if __name__ == '__main__':
    main()

"""
Cholec80 dataset preprocessing and annotation conversion.

Converts Cholec80 annotations to PTG format with action segments.

Usage:
    python scripts/preprocess_cholec80.py \
        --data_root D:/Datasets/cholec80 \
        --output_dir data/cholec80_annotations
"""
import json
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict


def parse_cholec80_phases(phase_file: Path) -> List[Dict]:
    """
    Parse Cholec80 phase annotations.
    
    Format: Each line is "frame_number phase_id"
    Phases: 0-6 (7 surgical phases)
    """
    segments = []
    current_phase = None
    start_frame = 0
    
    with open(phase_file) as f:
        lines = f.readlines()
    
    for frame_idx, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        
        phase_id = int(parts[1])
        
        if current_phase is None:
            # First frame
            current_phase = phase_id
            start_frame = frame_idx
        elif phase_id != current_phase:
            # Phase change
            segments.append({
                'action': f'phase_{current_phase}',
                'start_frame': start_frame,
                'end_frame': frame_idx - 1,
                'duration': frame_idx - start_frame,
                'is_chiral': False
            })
            current_phase = phase_id
            start_frame = frame_idx
    
    # Add last segment
    if current_phase is not None:
        segments.append({
            'action': f'phase_{current_phase}',
            'start_frame': start_frame,
            'end_frame': len(lines) - 1,
            'duration': len(lines) - start_frame,
            'is_chiral': False
        })
    
    return segments


def convert_to_ptg_format(
    video_id: str,
    segments: List[Dict],
    total_frames: int
) -> Dict:
    """Convert segments to PTG annotation format."""
    return {
        'video_id': video_id,
        'total_frames': total_frames,
        'segments': segments
    }


def main():
    parser = argparse.ArgumentParser(description='Preprocess Cholec80')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Cholec80 data root directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for annotations')
    parser.add_argument('--train_videos', type=int, nargs='+',
                       default=list(range(1, 41)),
                       help='Video IDs for training')
    parser.add_argument('--val_videos', type=int, nargs='+',
                       default=list(range(41, 61)),
                       help='Video IDs for validation')
    parser.add_argument('--test_videos', type=int, nargs='+',
                       default=list(range(61, 81)),
                       help='Video IDs for testing')
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    splits = {
        'train': args.train_videos,
        'val': args.val_videos,
        'test': args.test_videos
    }
    
    for split_name, video_ids in splits.items():
        print(f"\nProcessing {split_name} split ({len(video_ids)} videos)...")
        
        annotations = []
        
        for video_id in video_ids:
            # Find phase annotation file
            phase_file = data_root / 'phase_annotations' / f'video{video_id:02d}-phase.txt'
            
            if not phase_file.exists():
                print(f"  Warning: {phase_file} not found, skipping")
                continue
            
            # Parse annotations
            segments = parse_cholec80_phases(phase_file)
            
            # Get total frames
            total_frames = segments[-1]['end_frame'] + 1 if segments else 0
            
            # Convert to PTG format
            annotation = convert_to_ptg_format(
                video_id=f'video{video_id:02d}',
                segments=segments,
                total_frames=total_frames
            )
            
            annotations.append(annotation)
        
        # Save to JSON
        output_file = output_dir / f'{split_name}_annotations.json'
        with open(output_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        print(f"  Saved {len(annotations)} annotations to {output_file}")
    
    print(f"\n✓ Cholec80 preprocessing complete!")
    print(f"  Annotations saved to {output_dir}")


if __name__ == '__main__':
    main()

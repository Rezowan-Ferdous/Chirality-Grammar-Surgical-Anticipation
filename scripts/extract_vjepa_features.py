"""
V-JEPA feature extraction for Cholec80.

Extracts frozen V-JEPA2 features from surgical videos and saves as .npy files.

Usage:
    python scripts/extract_vjepa_features.py \
        --video_dir D:/Datasets/cholec80/videos \
        --output_dir data/features/cholec80 \
        --num_frames 16
"""
import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.vjepa_extractor import VJEPAExtractor


def extract_frames_from_video(
    video_path: Path,
    num_frames: int = 16,
    target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Extract uniformly sampled frames from video.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample
        target_size: Target frame size (H, W)
        
    Returns:
        (num_frames, H, W, 3) array
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        raise ValueError(f"Video has no frames: {video_path}")
    
    # Sample frame indices uniformly
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize
            frame = cv2.resize(frame, target_size)
            frames.append(frame)
        else:
            # Fallback: use last valid frame
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((*target_size, 3), dtype=np.uint8))
    
    cap.release()
    
    return np.stack(frames)  # (num_frames, H, W, 3)


def main():
    parser = argparse.ArgumentParser(description='Extract V-JEPA features')
    parser.add_argument('--video_dir', type=str, required=True,
                       help='Directory with video files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for features')
    parser.add_argument('--num_frames', type=int, default=16,
                       help='Number of frames to sample per video')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for processing')
    parser.add_argument('--model_name', type=str, 
                       default='facebook/vjepa2-vitl-fpc64-256',
                       help='V-JEPA model name')
    args = args.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load V-JEPA model
    print(f"\nLoading V-JEPA model: {args.model_name}")
    extractor = VJEPAExtractor(
        model_name=args.model_name,
        freeze_backbone=True,
        output_dim=512
    ).to(device)
    extractor.eval()
    
    # Find all video files
    video_dir = Path(args.video_dir)
    video_files = list(video_dir.glob('*.mp4')) + list(video_dir.glob('*.avi'))
    
    print(f"\nFound {len(video_files)} videos in {video_dir}")
    
    # Process each video
    for video_path in tqdm(video_files, desc="Extracting features"):
        video_id = video_path.stem
        output_path = output_dir / f"{video_id}_vjepa.npy"
        
        # Skip if already processed
        if output_path.exists():
            continue
        
        try:
            # Extract frames
            frames = extract_frames_from_video(
                video_path,
                num_frames=args.num_frames
            )  # (T, H, W, 3)
            
            # Convert to tensor
            frames_tensor = torch.from_numpy(frames).float() / 255.0  # Normalize
            frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # (T, 3, H, W)
            frames_tensor = frames_tensor.unsqueeze(0).to(device)  # (1, T, 3, H, W)
            
            # Extract features
            with torch.no_grad():
                features = extractor(frames_tensor)  # (1, T, 512)
            
            # Save
            features_np = features.squeeze(0).cpu().numpy()  # (T, 512)
            np.save(output_path, features_np)
            
        except Exception as e:
            print(f"\nError processing {video_path}: {e}")
            continue
    
    print(f"\n✓ Feature extraction complete! Features saved to {output_dir}")


if __name__ == '__main__':
    from typing import Tuple
    main()

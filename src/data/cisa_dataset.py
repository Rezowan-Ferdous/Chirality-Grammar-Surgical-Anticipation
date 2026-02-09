"""
CiSA (Chirality in Surgical Actions) Dataset.

Unified meta-dataset with chirality annotations from:
- Cholec80
- CholecT50
- JIGSAWS
- SAR-RARP50

Paper: "We construct a chirality-aware action recognition benchmark
by unifying action labels across surgical video datasets"
"""
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional
import numpy as np
from pathlib import Path
import json


class CiSADataset(Dataset):
    """
    Chirality in Surgical Actions benchmark.
    
    Supports variable observation and prediction rates for long-horizon anticipation.
    """
    
    def __init__(
        self,
        data_root: str,
        dataset_name: str,  # 'cholec80', 'cholect50', 'jigsaws', 'sar-rarp50'
        split: str = 'train',
        observation_rate: float = 0.5,
        prediction_rate: float = 0.2,
        feature_dir: Optional[str] = None,
        use_precomputed_features: bool = True,
        num_frames: int = 16
    ):
        """
        Args:
            data_root: Root directory for datasets
            dataset_name: Name of dataset
            split: 'train', 'val', or 'test'
            observation_rate: Fraction of video to observe (α)
            prediction_rate: Fraction of video to predict (β)
            feature_dir: Directory with pre-extracted V-JEPA features
            use_precomputed_features: If True, load features, else load raw frames
            num_frames: Number of frames to sample for V-JEPA
        """
        self.data_root = Path(data_root)
        self.dataset_name = dataset_name.lower()
        self.split = split
        self.observation_rate = observation_rate
        self.prediction_rate = prediction_rate
        self.feature_dir = Path(feature_dir) if feature_dir else None
        self.use_precomputed_features = use_precomputed_features
        self.num_frames = num_frames
        
        # Load annotations
        self.annotations = self._load_annotations()
        
        # Action vocabulary (unified across datasets)
        self.action_vocab = self._build_action_vocab()
        self.action_to_idx = {a: i for i, a in enumerate(self.action_vocab)}
        
        print(f"CiSA {dataset_name} ({split}): {len(self.annotations)} videos")
        print(f"  Observation rate: {observation_rate:.1%}")
        print(f"  Prediction rate: {prediction_rate:.1%}")
        print(f"  Action vocabulary: {len(self.action_vocab)}")
    
    def _load_annotations(self) -> List[Dict]:
        """Load dataset annotations with chirality labels."""
        anno_path = self.data_root / self.dataset_name / f"{self.split}_annotations.json"
        
        if not anno_path.exists():
            # Fallback: create dummy data for testing
            print(f"Warning: {anno_path} not found, generating dummy data")
            return self._generate_dummy_annotations()
        
        with open(anno_path) as f:
            annotations = json.load(f)
        
        return annotations
    
    def _generate_dummy_annotations(self) -> List[Dict]:
        """Generate dummy annotations for testing."""
        annotations = []
        
        for i in range(10):  # 10 dummy videos
            video_id = f"video_{i:03d}"
            total_frames = np.random.randint(1000, 3000)
            
            # Generate action segments
            segments = []
            current_frame = 0
            
            while current_frame < total_frames:
                action = np.random.choice(['pick', 'grasp', 'pull', 'push', 'cut', 'drop'])
                duration = np.random.randint(50, 200)
                end_frame = min(current_frame + duration, total_frames)
                
                segments.append({
                    'action': action,
                    'start_frame': current_frame,
                    'end_frame': end_frame,
                    'duration': end_frame - current_frame,
                    'objects': ['tissue', 'instrument'],
                    'is_chiral': action in ['push', 'pull', 'pick', 'drop']
                })
                
                current_frame = end_frame + 1
            
            annotations.append({
                'video_id': video_id,
                'total_frames': total_frames,
                'segments': segments
            })
        
        return annotations
    
    def _build_action_vocab(self) -> List[str]:
        """Build unified action vocabulary."""
        # Collect all unique actions
        actions = set()
        for anno in self.annotations:
            for seg in anno['segments']:
                actions.add(seg['action'])
        
        return sorted(list(actions))
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training sample.
        
        Returns:
            Dictionary with:
                - obs_features: (T_obs, D) observed features
                - future_actions: (T_fut,) future action labels
                - future_durations: (T_fut,) future durations
                - future_objects: (T_fut, num_objects) future object states
                - chirality_mask: (T_fut,) mask for chiral actions
        """
        anno = self.annotations[idx]
        total_frames = anno['total_frames']
        segments = anno['segments']
        
        # Compute observation and prediction boundaries
        obs_end = int(total_frames * self.observation_rate)
        pred_end = int(total_frames * (self.observation_rate + self.prediction_rate))
        
        # Extract observed segments
        obs_segments = [s for s in segments if s['start_frame'] < obs_end]
        
        # Extract future segments (to predict)
        future_segments = [s for s in segments 
                          if s['start_frame'] >= obs_end and s['end_frame'] <= pred_end]
        
        # Load or generate observation features
        if self.use_precomputed_features and self.feature_dir:
            obs_features = self._load_precomputed_features(anno['video_id'], obs_end)
        else:
            obs_features = torch.randn(self.num_frames, 512)  # Placeholder
        
        # Prepare future targets
        future_actions = torch.tensor([
            self.action_to_idx[s['action']] for s in future_segments
        ], dtype=torch.long)
        
        future_durations = torch.tensor([
            s['duration'] for s in future_segments
        ], dtype=torch.float)
        
        # Object states (placeholder - binary multi-label)
        num_objects = 16
        future_objects = torch.zeros(len(future_segments), num_objects)
        for i, seg in enumerate(future_segments):
            # Dummy: random objects
            obj_indices = np.random.choice(num_objects, size=2, replace=False)
            future_objects[i, obj_indices] = 1.0
        
        # Chirality mask
        chirality_mask = torch.tensor([
            1.0 if s.get('is_chiral', False) else 0.0
            for s in future_segments
        ], dtype=torch.float)
        
        return {
            'obs_features': obs_features,
            'future_actions': future_actions,
            'future_durations': future_durations,
            'future_objects': future_objects,
            'chirality_mask': chirality_mask,
            'video_id': anno['video_id']
        }
    
    def _load_precomputed_features(
        self,
        video_id: str,
        obs_end: int
    ) -> torch.Tensor:
        """Load pre-extracted V-JEPA features."""
        feature_path = self.feature_dir / f"{video_id}_vjepa.npy"
        
        if feature_path.exists():
            features = np.load(feature_path)
            # Sample features up to obs_end
            features = torch.from_numpy(features[:obs_end])
        else:
            # Fallback: random features
            features = torch.randn(self.num_frames, 512)
        
        return features.float()


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for variable-length sequences.
    
    Pads to maximum length in batch.
    """
    # Find max lengths
    max_obs_len = max(item['obs_features'].shape[0] for item in batch)
    max_fut_len = max(item['future_actions'].shape[0] for item in batch)
    
    B = len(batch)
    D = batch[0]['obs_features'].shape[1]  # Feature dimension
    num_objects = batch[0]['future_objects'].shape[1]
    
    # Initialize tensors
    obs_features = torch.zeros(B, max_obs_len, D)
    future_actions = torch.full((B, max_fut_len), -100, dtype=torch.long)  # -100 = ignore
    future_durations = torch.zeros(B, max_fut_len)
    future_objects = torch.zeros(B, max_fut_len, num_objects)
    chirality_mask = torch.zeros(B, max_fut_len)
    obs_mask = torch.zeros(B, max_obs_len, dtype=torch.bool)
    fut_mask = torch.zeros(B, max_fut_len, dtype=torch.bool)
    
    # Fill tensors
    for i, item in enumerate(batch):
        obs_len = item['obs_features'].shape[0]
        fut_len = item['future_actions'].shape[0]
        
        obs_features[i, :obs_len] = item['obs_features']
        future_actions[i, :fut_len] = item['future_actions']
        future_durations[i, :fut_len] = item['future_durations']
        future_objects[i, :fut_len] = item['future_objects']
        chirality_mask[i, :fut_len] = item['chirality_mask']
        obs_mask[i, obs_len:] = True  # Mask padding
        fut_mask[i, fut_len:] = True
    
    return {
        'obs_features': obs_features,
        'future_actions': future_actions,
        'future_durations': future_durations,
        'future_objects': future_objects,
        'chirality_mask': chirality_mask,
        'obs_mask': obs_mask,
        'fut_mask': fut_mask
    }


# Example usage
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    
    # Create dataset
    dataset = CiSADataset(
        data_root="D:/Datasets/",
        dataset_name="cholec80",
        split="train",
        observation_rate=0.5,
        prediction_rate=0.2,
        use_precomputed_features=False
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Test batch
    batch = next(iter(dataloader))
    print("Batch contents:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")

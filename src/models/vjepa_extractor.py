"""
V-JEPA feature extractor for PTG.

Extracts frozen self-supervised video features from V-JEPA2 backbone.

Paper: "V-JEPA2-ViTL/16 (frozen, 1024-d) + Linear projection (512-d)"
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import AutoModel, AutoImageProcessor


class VJEPAExtractor(nn.Module):
    """
    V-JEPA2 backbone for video feature extraction.
    
    Paper: "We use vjepa2-vitl-fpc64-256 (ViT-L/16) which encodes 
    multi-frame clips into a 1024-d feature vector"
    """
    
    def __init__(
        self,
        model_name: str = "facebook/vjepa2-vitl-fpc64-256",
        freeze_backbone: bool = True,
        output_dim: int = 512
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            freeze_backbone: If True, freeze V-JEPA weights
            output_dim: Dimension after linear projection (paper uses 512)
        """
        super().__init__()
        
        # Load V-JEPA2 backbone
        print(f"Loading V-JEPA2 model: {model_name}")
        self.vjepa = AutoModel.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.vjepa.parameters():
                param.requires_grad = False
            print("V-JEPA2 backbone frozen")
        
        # Get embedding dimension
        self.vjepa_dim = self.vjepa.config.hidden_size  # 1024 for ViT-L
        
        # Linear projection to output_dim
        self.projection = nn.Linear(self.vjepa_dim, output_dim)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(output_dim)
        
        print(f"V-JEPA dim: {self.vjepa_dim}, Output dim: {output_dim}")
    
    def forward(
        self,
        video_frames: torch.Tensor,
        return_raw: bool = False
    ) -> torch.Tensor:
        """
        Extract features from video frames.
        
        Args:
            video_frames: (B, T, C, H, W) or (B, T, 3, 224, 224)
            return_raw: If True, return raw V-JEPA features without projection
            
        Returns:
            (B, T, output_dim) if projected, else (B, T, vjepa_dim)
        """
        B, T, C, H, W = video_frames.shape
        
        # Reshape to (B*T, C, H, W) for batch processing
        frames_flat = video_frames.view(B * T, C, H, W)
        
        # Extract V-JEPA features
        with torch.set_grad_enabled(not self.vjepa.training):
            outputs = self.vjepa(pixel_values=frames_flat)
            # Get CLS token or mean pool
            if hasattr(outputs, 'last_hidden_state'):
                features = outputs.last_hidden_state.mean(dim=1)  # (B*T, vjepa_dim)
            else:
                features = outputs.pooler_output  # (B*T, vjepa_dim)
        
        # Reshape to (B, T, vjepa_dim)
        features = features.view(B, T, -1)
        
        if return_raw:
            return features
        
        # Apply projection
        features = self.projection(features)  # (B, T, output_dim)
        features = self.layer_norm(features)
        
        return features
    
    def extract_from_video_path(
        self,
        video_path: str,
        num_frames: int = 16,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """
        Extract features from video file.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to sample
            device: Device to run on
            
        Returns:
            (1, num_frames, output_dim) tensor
        """
        import cv2
        import numpy as np
        
        # Load video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames uniformly
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        
        # Preprocess frames
        frames = np.stack(frames)  # (T, H, W, 3)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (T, 3, H, W)
        frames = frames.unsqueeze(0).to(device)  # (1, T, 3, H, W)
        
        # Normalize (V-JEPA expects normalized inputs)
        frames = frames.float() / 255.0
        
        # Extract features
        with torch.no_grad():
            features = self.forward(frames)
        
        return features


class TemporalPooling(nn.Module):
    """
    Temporal pooling strategies for variable-length sequences.
    """
    
    def __init__(self, pooling_type: str = 'mean'):
        """
        Args:
            pooling_type: 'mean', 'max', 'last', or 'attention'
        """
        super().__init__()
        self.pooling_type = pooling_type
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Pool temporal dimension.
        
        Args:
            features: (B, T, D)
            
        Returns:
            (B, D)
        """
        if self.pooling_type == 'mean':
            return features.mean(dim=1)
        elif self.pooling_type == 'max':
            return features.max(dim=1)[0]
        elif self.pooling_type == 'last':
            return features[:, -1]
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")


# Example usage
if __name__ == '__main__':
    # Create extractor
    extractor = VJEPAExtractor(
        model_name="facebook/vjepa2-vitl-fpc64-256",
        freeze_backbone=True,
        output_dim=512
    )
    
    # Test forward pass
    B, T, C, H, W = 2, 16, 3, 224, 224
    video_frames = torch.randn(B, T, C, H, W)
    
    features = extractor(video_frames)
    print(f"Input shape: {video_frames.shape}")
    print(f"Output shape: {features.shape}")  # (2, 16, 512)
    
    # Test pooling
    pooler = TemporalPooling('mean')
    pooled = pooler(features)
    print(f"Pooled shape: {pooled.shape}")  # (2, 512)

"""
GLM Platform - Video Generator Trainer
========================================

Training module for video generation models including:
- Text-to-video generation
- Video diffusion models
- Temporal consistency models
"""

import os
import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from base_trainer import BaseTrainer, TrainingMetrics

logger = logging.getLogger('GLM-Training.Video')


class VideoDataset(Dataset):
    """Dataset for video generation training."""
    
    def __init__(
        self,
        data_path: str,
        num_frames: int = 16,
        frame_size: int = 256,
        mode: str = 'train',
    ):
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.mode = mode
        
        # Load video metadata or paths
        self.samples = []
        
        path = Path(data_path)
        if path.is_dir():
            # Look for video files or image sequences
            for ext in ['.mp4', '.avi', '.webm', '.mov']:
                for vid_file in path.glob(f'**/*{ext}'):
                    self.samples.append({
                        'path': vid_file,
                        'type': 'video',
                    })
            
            # Also check for image sequence folders
            for seq_dir in path.iterdir():
                if seq_dir.is_dir() and seq_dir.name.startswith('seq_'):
                    images = sorted(list(seq_dir.glob('*.*')))
                    if len(images) >= num_frames:
                        self.samples.append({
                            'path': seq_dir,
                            'type': 'sequence',
                        })
        
        logger.info(f"Loaded {len(self.samples)} video samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def load_video_frames(self, sample: Dict) -> torch.Tensor:
        """Load frames from video file or image sequence."""
        try:
            from PIL import Image
            import torchvision.transforms as T
            
            transform = T.Compose([
                T.Resize((self.frame_size, self.frame_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            
            frames = []
            
            if sample['type'] == 'video':
                try:
                    import cv2
                    cap = cv2.VideoCapture(str(sample['path']))
                    
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    indices = [int(i * total_frames / self.num_frames) for i in range(self.num_frames)]
                    
                    for idx in indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap.read()
                        if ret:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame = Image.fromarray(frame)
                            frames.append(transform(frame))
                        else:
                            frames.append(torch.zeros(3, self.frame_size, self.frame_size))
                    
                    cap.release()
                except Exception:
                    frames = [torch.zeros(3, self.frame_size, self.frame_size)] * self.num_frames
            
            elif sample['type'] == 'sequence':
                images = sorted(list(sample['path'].glob('*.*')))[:self.num_frames]
                
                for img_path in images:
                    img = Image.open(img_path).convert('RGB')
                    frames.append(transform(img))
                
                while len(frames) < self.num_frames:
                    frames.append(torch.zeros(3, self.frame_size, self.frame_size))
            
            return torch.stack(frames)  # [T, C, H, W]
        
        except Exception as e:
            logger.warning(f"Failed to load video: {e}")
            return torch.zeros(self.num_frames, 3, self.frame_size, self.frame_size)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        frames = self.load_video_frames(sample)
        
        return {
            'frames': frames,
            'caption': sample.get('caption', ''),
        }


class SpatialTemporalAttention(nn.Module):
    """Spatial-temporal attention block."""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, T * N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Spatial-temporal attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, T, N, C)
        x = self.proj(x)
        
        return x


class VideoDiffusionModel(nn.Module):
    """Video diffusion model with spatial-temporal attention."""
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        d_model: int = 320,
        num_layers: int = 12,
        num_heads: int = 8,
        num_frames: int = 16,
        frame_size: int = 256,
    ):
        super().__init__()
        
        self.num_frames = num_frames
        self.frame_size = frame_size
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(128, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        
        # Input projection
        self.input_proj = nn.Conv3d(in_channels, d_model, kernel_size=3, padding=1)
        
        # Spatial-temporal blocks
        self.blocks = nn.ModuleList([
            STBlock(d_model, num_heads) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Conv3d(d_model, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x, timesteps):
        """
        Args:
            x: [B, C, T, H, W] - Noisy video frames
            timesteps: [B] - Diffusion timesteps
        """
        B = x.size(0)
        
        # Time embedding
        t_emb = self._get_time_embedding(timesteps)
        t_emb = self.time_embed(t_emb)
        
        # Input projection
        h = self.input_proj(x)
        
        # Apply blocks
        for block in self.blocks:
            h = block(h, t_emb)
        
        # Output
        output = self.output_proj(h)
        
        return output
    
    def _get_time_embedding(self, timesteps):
        """Sinusoidal time embedding."""
        half_dim = 64
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class STBlock(nn.Module):
    """Spatial-Temporal Transformer Block."""
    
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SpatialTemporalAttention(d_model, num_heads)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
    
    def forward(self, x, t_emb=None):
        B, C, T, H, W = x.shape
        N = H * W
        
        # Reshape for attention: [B, T, N, C]
        x_flat = x.flatten(-2).permute(0, 2, 3, 1)
        
        # Attention with residual
        residual = x_flat
        x_flat = self.norm1(x_flat)
        x_flat = self.attn(x_flat)
        x_flat = x_flat + residual
        
        # MLP with residual
        residual = x_flat
        x_flat = self.norm2(x_flat)
        x_flat = self.mlp(x_flat) + residual
        
        # Reshape back: [B, C, T, H, W]
        x_out = x_flat.permute(0, 3, 1, 2).reshape(B, C, T, H, W)
        
        return x_out


class VideoGeneratorTrainer(BaseTrainer):
    """Trainer for video generation models."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.model = None
        self.train_loader = None
        self.val_loader = None
        
        # Model config
        video_config = config.get('video_generator', {})
        self.num_frames = video_config.get('num_frames', 16)
        self.frame_size = video_config.get('frame_size', 256)
        self.d_model = video_config.get('d_model', 320)
    
    def prepare_data(self):
        """Prepare video datasets."""
        logger.info("Preparing video data...")
        
        dataset_config = self.config.get('dataset', {})
        data_path = dataset_config.get('path', './data/videos')
        
        train_dataset = VideoDataset(
            data_path=data_path,
            num_frames=self.num_frames,
            frame_size=self.frame_size,
            mode='train',
        )
        
        val_dataset = VideoDataset(
            data_path=data_path,
            num_frames=self.num_frames,
            frame_size=self.frame_size,
            mode='val',
        )
        
        hp = self.config.get('hyperparameters', {})
        batch_size = hp.get('batch_size', 2)  # Small batch due to memory
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
    
    def build_model(self):
        """Build video generation model."""
        logger.info("Building video generation model...")
        
        self.model = VideoDiffusionModel(
            d_model=self.d_model,
            num_frames=self.num_frames,
            frame_size=self.frame_size,
        ).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Total parameters: {total_params:,}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        num_batches = 0
        
        for step, batch in enumerate(self.train_loader):
            frames = batch['frames'].to(self.device)  # [B, T, C, H, W]
            
            # Rearrange to [B, C, T, H, W]
            frames = frames.permute(0, 2, 1, 3, 4)
            
            # Sample noise and timesteps
            noise = torch.randn_like(frames)
            timesteps = torch.randint(0, 1000, (frames.size(0),), device=self.device)
            
            # Add noise (forward diffusion process)
            noisy_frames = self._add_noise(frames, noise, timesteps)
            
            # Predict noise
            predicted_noise = self.model(noisy_frames, timesteps)
            
            # Compute loss (simple MSE on noise prediction)
            loss = F.mse_loss(predicted_noise, noise)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            if step % self.logging_config.log_every_n_steps == 0:
                logger.info(f"Step {step}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / max(num_batches, 1)
        return {'loss': avg_loss}
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                frames = batch['frames'].to(self.device).permute(0, 2, 1, 3, 4)
                noise = torch.randn_like(frames)
                timesteps = torch.randint(0, 1000, (frames.size(0),), device=self.device)
                
                noisy_frames = self._add_noise(frames, noise, timesteps)
                predicted_noise = self.model(noisy_frames, timesteps)
                
                loss = F.mse_loss(predicted_noise, noise)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        return {'val_loss': avg_loss}
    
    def _add_noise(self, x, noise, timesteps):
        """Add noise to input based on diffusion schedule."""
        # Simple linear schedule
        sqrt_alpha_cumprod = 1 - (timesteps.float() / 1000.0).unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        
        noisy_x = sqrt_alpha_cumprod * x + (1 - sqrt_alpha_cumprod ** 2).sqrt() * noise
        
        return noisy_x
    
    def export(self, export_config: Dict[str, Any]):
        """Export video generation model."""
        logger.info("Exporting video generation model...")
        
        export_dir = self.output_dir / 'exports'
        export_dir.mkdir(exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': {
                'num_frames': self.num_frames,
                'frame_size': self.frame_size,
                'd_model': self.d_model,
            },
        }, export_dir / 'video_gen_model.pt')
        
        logger.info(f"Video model exported to {export_dir}")


# Import math for time embedding
import math

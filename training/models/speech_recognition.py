"""
GLM Platform - Speech Recognition Trainer
==========================================

Training module for Automatic Speech Recognition (ASR) models.
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

logger = logging.getLogger('GLM-Training.Speech')


class AudioDataset(Dataset):
    """Dataset for speech recognition training."""
    
    def __init__(
        self,
        data_path: str,
        sample_rate: int = 16000,
        max_duration: float = 30.0,
        mode: str = 'train',
    ):
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.mode = mode
        
        # Load manifest file or directory
        self.samples = []
        
        path = Path(data_path)
        if path.suffix in ['.json', '.jsonl']:
            import json
            with open(path, 'r') as f:
                if path.suffix == '.jsonl':
                    for line in f:
                        self.samples.append(json.loads(line.strip()))
                else:
                    self.samples = json.load(f)
        
        logger.info(f"Loaded {len(self.samples)} audio samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def load_audio(self, path: str) -> torch.Tensor:
        """Load and preprocess audio file."""
        try:
            import torchaudio
            
            waveform, sr = torchaudio.load(path)
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            return waveform.squeeze(0)
        except Exception as e:
            logger.warning(f"Failed to load audio {path}: {e}")
            return torch.zeros(self.sample_rate * 10)  # Return silence
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        waveform = self.load_audio(sample.get('audio_path', sample.get('path', '')))
        
        # Truncate or pad
        max_samples = int(self.max_duration * self.sample_rate)
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]
        else:
            waveform = F.pad(waveform, (0, max_samples - len(waveform)))
        
        return {
            'audio': waveform,
            'text': sample.get('text', sample.get('transcription', '')),
            'duration': min(len(waveform) / self.sample_rate, self.max_duration),
        }


class ConformerEncoder(nn.Module):
    """Conformer encoder for speech recognition."""
    
    def __init__(
        self,
        input_dim: int = 80,
        d_model: int = 144,
        num_layers: int = 16,
        num_heads: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        
        self.conformer_layers = nn.ModuleList([
            ConformerBlock(d_model, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, lengths=None):
        x = self.input_proj(x)
        
        for layer in self.conformer_layers:
            x = layer(x)
        
        return x


class ConformerBlock(nn.Module):
    """Conformer block with multi-head attention and convolutions."""
    
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        
        # Feed-forward module 1
        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        
        # Convolution module
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(d_model, d_model, kernel_size=7, padding=3, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout),
        )
        self.conv_norm = nn.LayerNorm(d_model)
        
        # Feed-forward module 2
        self.ffn2 = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        
        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # FFN 1 + residual
        x = x + 0.5 * self.ffn1(x)
        
        # Self-attention + residual
        x_norm = self.attn_norm(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)
        
        # Convolution + residual
        x_norm = self.conv_norm(x)
        conv_out = self.conv(x_norm.transpose(1, 2)).transpose(1, 2)
        x = x + self.dropout(conv_out)
        
        # FFN 2 + residual
        x = x + 0.5 * self.ffn2(x)
        
        # Final norm
        x = self.final_norm(x)
        
        return x


class SpeechRecognitionTrainer(BaseTrainer):
    """Trainer for speech recognition models."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.model = None
        self.train_loader = None
        self.val_loader = None
        
        # Model config
        speech_config = config.get('speech_recognition', {})
        self.vocab_size = speech_config.get('vocab_size', 5000)
        self.d_model = speech_config.get('d_model', 144)
        self.num_layers = speech_config.get('num_layers', 16)
    
    def prepare_data(self):
        """Prepare speech datasets."""
        logger.info("Preparing speech recognition data...")
        
        dataset_config = self.config.get('dataset', {})
        data_path = dataset_config.get('path', './data/speech')
        
        train_dataset = AudioDataset(
            data_path=data_path,
            mode='train',
        )
        
        val_dataset = AudioDataset(
            data_path=data_path,
            mode='val',
        )
        
        hp = self.config.get('hyperparameters', {})
        batch_size = hp.get('batch_size', 8)
        
        collate_fn = self._collate_fn
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
        )
    
    def _collate_fn(self, batch):
        """Collate function for variable-length sequences."""
        audios = [item['audio'] for item in batch]
        texts = [item['text'] for item in batch]
        
        # Pad audios to same length
        max_len = max(a.size(0) for a in audios)
        padded_audios = torch.stack([
            F.pad(a, (0, max_len - a.size(0))) for a in audios
        ])
        
        return {
            'audio': padded_audios,
            'text': texts,
        }
    
    def build_model(self):
        """Build speech recognition model."""
        logger.info("Building speech recognition model...")
        
        self.encoder = ConformerEncoder(
            input_dim=80,  # MFCC features
            d_model=self.d_model,
            num_layers=self.num_layers,
        )
        
        # Decoder (CTC or attention-based)
        self.decoder = nn.Linear(self.d_model, self.vocab_size)
        
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        
        total_params = sum(p.numel() for p in self.encoder.parameters()) + \
                       sum(p.numel() for p in self.decoder.parameters())
        logger.info(f"Total parameters: {total_params:,}")
    
    def extract_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract acoustic features from raw audio."""
        try:
            import torchaudio.transforms as T
            
            # Compute MFCC features
            mfcc_transform = T.MFCC(
                sample_rate=16000,
                n_mfcc=80,
                melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 128},
            )
            
            features = mfcc_transform(audio)
            return features.transpose(1, 2)  # [batch, time, features]
        except Exception:
            # Fallback: use raw audio as features
            return audio.unsqueeze(-1).expand(-1, -1, 80)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.encoder.train()
        self.decoder.train()
        
        total_loss = 0
        num_batches = 0
        
        criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        
        for step, batch in enumerate(self.train_loader):
            audio = batch['audio'].to(self.device)
            texts = batch['text']
            
            # Extract features
            features = self.extract_features(audio)
            
            # Encode
            encoded = self.encoder(features)
            
            # Decode
            logits = self.decoder(encoded)
            
            # Create targets (simplified - would need proper tokenizer)
            target_lengths = torch.tensor([min(len(t), 100) for t in texts])
            input_lengths = torch.full((logits.size(0),), logits.size(1), dtype=torch.long)
            
            # Dummy targets for demonstration
            targets = torch.randint(1, self.vocab_size, (sum(target_lengths.tolist()),)).to(self.device)
            
            loss = criterion(
                logits.log_softmax(-1).transpose(0, 1),
                targets,
                input_lengths,
                target_lengths,
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.decoder.parameters()), 1.0
            )
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        return {'loss': avg_loss}
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        self.encoder.eval()
        self.decoder.eval()
        
        total_loss = 0
        num_batches = 0
        
        criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        
        with torch.no_grad():
            for batch in self.val_loader:
                audio = batch['audio'].to(self.device)
                
                features = self.extract_features(audio)
                encoded = self.encoder(features)
                logits = self.decoder(encoded)
                
                target_lengths = torch.tensor([100] * audio.size(0))  # Simplified
                input_lengths = torch.full((logits.size(0),), logits.size(1), dtype=torch.long)
                targets = torch.randint(1, self.vocab_size, (sum(target_lengths.tolist()),)).to(self.device)
                
                loss = criterion(logits.log_softmax(-1).transpose(0, 1), targets, input_lengths, target_lengths)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Calculate approximate WER (would need proper decoding)
        wer = avg_loss * 10  # Placeholder
        
        return {'val_loss': avg_loss, 'wer': wer}
    
    def export(self, export_config: Dict[str, Any]):
        """Export speech recognition model."""
        logger.info("Exporting speech recognition model...")
        
        export_dir = self.output_dir / 'exports'
        export_dir.mkdir(exist_ok=True)
        
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'config': {
                'd_model': self.d_model,
                'vocab_size': self.vocab_size,
                'num_layers': self.num_layers,
            },
        }, export_dir / 'asr_model.pt')
        
        logger.info(f"ASR model exported to {export_dir}")

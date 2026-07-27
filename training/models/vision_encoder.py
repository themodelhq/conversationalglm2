"""
GLM Platform - Vision Encoder Trainer
======================================

Training module for vision encoder models including:
- Image classification
- Visual feature extraction
- Vision-language pretraining
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
from torchvision import transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from base_trainer import BaseTrainer, TrainingMetrics

logger = logging.getLogger('GLM-Training.Vision')


class VisionDataset(Dataset):
    """Dataset for vision model training."""
    
    def __init__(
        self,
        data_path: str,
        transform=None,
        mode: str = 'train',
        image_size: int = 224,
    ):
        self.transform = transform or self._default_transform(image_size, mode)
        self.mode = mode
        
        # Load data from directory or file
        self.samples = []
        path = Path(data_path)
        
        if path.is_dir():
            # Assume class folders structure
            classes = sorted([d.name for d in path.iterdir() if d.is_dir()])
            self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
            
            for cls in classes:
                cls_path = path / cls
                for img_file in cls_path.glob('*.*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                        self.samples.append({
                            'path': img_file,
                            'label': self.class_to_idx[cls],
                        })
        
        logger.info(f"Loaded {len(self.samples)} samples for {mode}")
    
    def _default_transform(self, image_size: int, mode: str):
        """Default image transforms."""
        if mode == 'train':
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        image = Image.open(sample['path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'label': torch.tensor(sample['label'], dtype=torch.long),
        }


class VisionEncoderTrainer(BaseTrainer):
    """Trainer for vision encoder models."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.image_size = config.get('vision', {}).get('image_size', 224)
        self.num_classes = config.get('vision', {}).get('num_classes', 1000)
        
        self.model = None
        self.train_loader = None
        self.val_loader = None
    
    def prepare_data(self):
        """Prepare vision datasets."""
        logger.info("Preparing vision data...")
        
        dataset_config = self.config.get('dataset', {})
        data_path = dataset_config.get('path', './data/images')
        val_split = dataset_config.get('validation_split', 0.1)
        
        # Create datasets
        train_dataset = VisionDataset(
            data_path=data_path,
            image_size=self.image_size,
            mode='train',
        )
        
        val_dataset = VisionDataset(
            data_path=data_path,
            image_size=self.image_size,
            mode='val',
        )
        
        hp = self.config.get('hyperparameters', {})
        batch_size = hp.get('batch_size', 32)
        
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
        """Build vision encoder model."""
        logger.info("Building vision encoder...")
        
        try:
            from transformers import ViTModel, ViTConfig, AutoModel
            
            # Try loading pretrained ViT
            model_name = self.config.get('vision', {}).get('model_name', 'google/vit-base-patch16-224')
            
            self.model = AutoModel.from_pretrained(model_name)
            
            # Add classification head
            self.classifier = nn.Linear(self.model.config.hidden_size, self.num_classes)
            
            # Move to device
            self.model = self.model.to(self.device)
            self.classifier = self.classifier.to(self.device)
            
        except Exception as e:
            logger.warning(f"Could not load pretrained model: {e}")
            logger.info("Using custom Vision Transformer")
            self._build_custom_vit()
    
    def _build_custom_vit(self):
        """Build a custom Vision Transformer."""
        from transformers import ViTConfig, ViTModel
        
        config = ViTConfig(
            image_size=self.image_size,
            patch_size=16,
            num_layers=12,
            num_heads=12,
            hidden_size=768,
            mlp_dim=3072,
            num_classes=self.num_classes,
        )
        
        self.model = ViTModel(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_classes)
        
        self.model = self.model.to(self.device)
        self.classifier = self.classifier.to(self.device)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.classifier.train()
        
        total_loss = 0
        correct = 0
        total = 0
        num_batches = 0
        
        criterion = nn.CrossEntropyLoss()
        
        for step, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            outputs = self.model(pixel_values=images)
            logits = self.classifier(outputs.last_hidden_state[:, 0])  # CLS token
            loss = criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.classifier.parameters()), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            num_batches += 1
            self.global_step += 1
        
        accuracy = correct / max(total, 1)
        avg_loss = total_loss / max(num_batches, 1)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
        }
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        self.classifier.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        num_batches = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(pixel_values=images)
                logits = self.classifier(outputs.last_hidden_state[:, 0])
                loss = criterion(logits, labels)
                
                total_loss += loss.item()
                _, predicted = logits.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                num_batches += 1
        
        accuracy = correct / max(total, 1)
        avg_loss = total_loss / max(num_batches, 1)
        
        return {
            'val_loss': avg_loss,
            'accuracy': accuracy,
        }
    
    def export(self, export_config: Dict[str, Any]):
        """Export vision model."""
        logger.info("Exporting vision encoder...")
        
        export_dir = self.output_dir / 'exports'
        export_dir.mkdir(exist_ok=True)
        
        formats = export_config.get('formats', ['pytorch'])
        
        for fmt in formats:
            if fmt in ['pytorch', 'safetensors']:
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'classifier_state_dict': self.classifier.state_dict(),
                }, export_dir / f'vision_model.{fmt}')
            
            self.tokenizer = None  # No tokenizer for pure vision
        
        logger.info(f"Vision model exported to {export_dir}")

"""
GLM Platform - Base Trainer Class
=================================

This module provides the base trainer class that all model-specific
trainers inherit from. It handles common training functionality like:
- Checkpoint management
- Logging and metrics tracking
- Distributed training setup
- Mixed precision training
- Early stopping
- Learning rate scheduling
"""

import os
import sys
import time
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger('GLM-Training')


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    epoch: int = 0
    step: int = 0
    total_steps: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    accuracy: Optional[float] = None
    perplexity: Optional[float] = None
    gpu_usage: List[float] = field(default_factory=list)
    gpu_memory: List[float] = field(default_factory=list)
    gpu_temperature: List[float] = field(default_factory=list)
    throughput: Optional[float] = None
    elapsed_time: float = 0.0
    eta: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'epoch': self.epoch,
            'step': self.step,
            'total_steps': self.total_steps,
            'loss': self.loss,
            'learning_rate': self.learning_rate,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'accuracy': self.accuracy,
            'perplexity': self.perplexity,
            'gpu_usage': self.gpu_usage,
            'gpu_memory': self.gpu_memory,
            'throughput': self.throughput,
            'elapsed_time': self.elapsed_time,
            'eta': self.eta,
        }


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint saving."""
    save_dir: str = 'checkpoints'
    save_every_n_steps: int = 1000
    save_every_n_epochs: int = 1
    keep_last_n: int = 5
    save_optimizer_state: bool = True
    save_best_only: bool = False


@dataclass 
class EarlyStoppingConfig:
    """Configuration for early stopping."""
    enabled: bool = True
    monitor: str = 'val_loss'
    patience: int = 5
    min_delta: float = 1e-4
    mode: str = 'min'  # 'min' or 'max'


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    log_dir: str = 'logs'
    log_every_n_steps: int = 10
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: str = 'glm-training'
    wandb_entity: Optional[str] = None


class BaseTrainer(ABC):
    """
    Base class for all GLM model trainers.
    
    Provides common functionality for:
    - Model initialization and configuration
    - Data loading and preprocessing
    - Training loop with validation
    - Checkpoint management
    - Metrics logging (TensorBoard, W&B)
    - Export functionality
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base trainer.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'output'))
        
        # Initialize configurations
        self.checkpoint_config = CheckpointConfig(
            **config.get('checkpoint', {})
        )
        self.early_stopping_config = EarlyStoppingConfig(
            **config.get('early_stopping', {'enabled': True})
        )
        self.logging_config = LoggingConfig(
            **config.get('logging', {'tensorboard': True})
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf') if self.early_stopping_config.mode == 'min' else float('-inf')
        self.patience_counter = 0
        self.start_time: Optional[datetime] = None
        
        # Metrics history
        self.metrics_history: List[TrainingMetrics] = []
        
        # Initialize components
        self._setup_logging()
        self._setup_device()
        
    @abstractmethod
    def prepare_data(self):
        """Prepare and load training data."""
        pass
    
    @abstractmethod
    def build_model(self):
        """Build the model architecture."""
        pass
    
    @abstractmethod
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        pass
    
    @abstractmethod
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        pass
    
    def _setup_logging(self):
        """Setup logging handlers and integrations."""
        log_dir = self.output_dir / self.logging_config.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard integration
        if self.logging_config.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=str(log_dir / 'tensorboard'))
                logger.info("TensorBoard logging initialized")
            except ImportError:
                logger.warning("TensorBoard not available")
                self.tb_writer = None
        
        # Weights & Biases integration
        if self.logging_config.wandb:
            try:
                import wandb
                wandb.init(
                    project=self.logging_config.wandb_project,
                    entity=self.logging_config.wandb_entity,
                    name=f"{self.config.get('model_type')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    config=self.config,
                )
                self.wandb_run = wandb
                logger.info("Weights & Biases logging initialized")
            except ImportError:
                logger.warning("Weights & Biases not available")
                self.wandb_run = None
        else:
            self.wandb_run = None
    
    def _setup_device(self):
        """Setup compute device(s)."""
        import torch
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        logger.info(f"Using device: {self.device}")
        if self.num_gpus > 0:
            logger.info(f"Available GPUs: {self.num_gpus}")
    
    def setup_distributed(self):
        """Setup distributed training environment."""
        import torch
        import torch.distributed as dist
        
        hardware_config = self.config.get('hardware', {})
        backend = hardware_config.get('distributed_backend', 'ddp')
        gpus = hardware_config.get('gpus', 1)
        
        if gpus <= 1:
            self.is_distributed = False
            return
        
        self.is_distributed = True
        
        # Environment variables for distributed training
        os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
        
        # Initialize process group
        if backend == 'deepspeed':
            import deepspeed
            self.deepspeed_initialized = True
            logger.info("DeepSpeed distributed training configured")
        elif backend == 'fsdp':
            # FSDP setup for PyTorch 2.x
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            self.fsdp_enabled = True
            logger.info("FSDP distributed training configured")
        else:
            # Default DDP
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=gpus,
                rank=int(os.environ.get('LOCAL_RANK', 0))
            )
            logger.info("DDP distributed training configured")
    
    def setup_mixed_precision(self):
        """Setup mixed precision training (AMP)."""
        hp = self.config.get('hyperparameters', {})
        self.use_mixed_precision = hp.get('mixed_precision', True)
        
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed precision training enabled (AMP)")
        else:
            self.scaler = None
            logger.info("Mixed precision training disabled")
    
    def setup_gradient_checkpointing(self):
        """Setup gradient checkpointing for memory efficiency."""
        hp = self.config.get('hyperparameters', {})
        self.use_gradient_checkpointing = hp.get('gradient_checkpointing', False)
        
        if self.use_gradient_checkpointing:
            logger.info("Gradient checkpointing enabled")
    
    def get_lr_scheduler(self, optimizer, total_steps: int):
        """Create learning rate scheduler based on configuration."""
        import torch
        from torch.optim.lr_scheduler import (
            CosineAnnealingLR,
            LinearLR,
            SequentialLR,
            LambdaLR
        )
        
        hp = self.config.get('hyperparameters', {})
        scheduler_type = hp.get('scheduler', 'cosine')
        warmup_steps = hp.get('warmup_steps', 100)
        learning_rate = hp.get('learning_rate', 1e-4)
        
        if scheduler_type == 'cosine':
            main_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max(total_steps - warmup_steps, 1),
                eta_min=learning_rate * 0.1
            )
        elif scheduler_type == 'linear':
            main_scheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=max(total_steps - warmup_steps, 1)
            )
        elif scheduler_type == 'constant':
            main_scheduler = LambdaLR(optimizer, lambda step: 1.0)
        else:
            main_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
        
        # Add warmup
        if warmup_steps > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_steps]
            )
        else:
            scheduler = main_scheduler
        
        return scheduler
    
    def get_optimizer(self, model_parameters):
        """Create optimizer based on configuration."""
        import torch
        from torch.optim import AdamW, Adam, SGD
        
        hp = self.config.get('hyperparameters', {})
        optimizer_type = hp.get('optimizer', 'adamw')
        learning_rate = hp.get('learning_rate', 1e-4)
        weight_decay = hp.get('weight_decay', 0.01)
        
        if optimizer_type == 'adamw':
            optimizer = AdamW(
                model_parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-6
            )
        elif optimizer_type == 'adamw8bit':
            try:
                import bitsandbytes as bnb
                optimizer = bnb.AdamW8bit(
                    model_parameters,
                    lr=learning_rate,
                    weight_decay=weight_decay
                )
            except ImportError:
                logger.warning("bitsandbytes not available, falling back to AdamW")
                optimizer = AdamW(model_parameters, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type == 'adam':
            optimizer = Adam(model_parameters, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            optimizer = SGD(model_parameters, lr=learning_rate, momentum=0.9)
        else:
            optimizer = AdamW(model_parameters, lr=learning_rate, weight_decay=weight_decay)
        
        logger.info(f"Optimizer: {optimizer_type} with lr={learning_rate}")
        return optimizer
    
    def log_metrics(self, metrics: TrainingMetrics, step: int):
        """Log metrics to all configured outputs."""
        # Log to console
        if step % self.logging_config.log_every_n_steps == 0:
            logger.info(
                f"Epoch {metrics.epoch}/{self.total_epochs} | "
                f"Step {metrics.step} | "
                f"Loss: {metrics.loss:.4f} | "
                f"LR: {metrics.learning_rate:.2e}" +
                (f" | Val Loss: {metrics.val_loss:.4f}" if metrics.val_loss else "") +
                (f" | ETA: {metrics.eta}" if metrics.eta else "")
            )
        
        # Log to TensorBoard
        if self.tb_writer:
            metric_dict = metrics.to_dict()
            for key, value in metric_dict.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(f'training/{key}', value, step)
            
            # Log GPU stats separately as histograms
            if metrics.gpu_usage:
                for i, usage in enumerate(metrics.gpu_usage):
                    self.tb_writer.add_scalar(f'gpu/{i}_usage', usage, step)
            if metrics.gpu_memory:
                for i, mem in enumerate(metrics.gpu_memory):
                    self.tb_writer.add_scalar(f'gpu/{i}_memory', mem, step)
            
            self.tb_writer.flush()
        
        # Log to W&B
        if self.wandb_run:
            self.wandb_run.log(metrics.to_dict(), step=step)
        
        # Store in history
        self.metrics_history.append(metrics)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save a training checkpoint."""
        import torch
        
        checkpoint_dir = self.output_dir / self.checkpoint_config.save_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'best_metric': self.best_metric,
            'metrics_history': [m.to_dict() for m in self.metrics_history[-100:]],  # Last 100
            'timestamp': datetime.now().isoformat(),
        }
        
        if self.checkpoint_config.save_optimizer_state and hasattr(self, 'optimizer'):
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        if hasattr(self, 'scheduler'):
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save checkpoint file
        filename = f'checkpoint-epoch-{epoch}-step-{self.global_step}.pt'
        filepath = checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint: {filepath}")
        
        # Save as latest
        latest_path = checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)
        
        # Save best model
        if is_best:
            best_path = checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model checkpoint")
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints(checkpoint_dir)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load a training checkpoint."""
        import torch
        
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Restore state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_metric = checkpoint.get('best_metric', float('inf'))
        
        if 'model_state_dict' in checkpoint and hasattr(self, 'model'):
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint and hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and hasattr(self, 'scheduler'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Restored from epoch {self.current_epoch}, step {self.global_step}")
    
    def _cleanup_checkpoints(self, checkpoint_dir: Path):
        """Remove old checkpoints beyond keep limit."""
        if self.checkpoint_config.keep_last_n <= 0:
            return
        
        checkpoints = sorted(checkpoint_dir.glob('checkpoint-*.pt'), key=lambda p: p.stat().st_mtime)
        
        # Keep only recent checkpoints
        while len(checkpoints) > self.checkpoint_config.keep_last_n + 2:  # +2 for latest and best
            old_checkpoint = checkpoints.pop(0)
            if old_checkpoint.name not in ['latest.pt', 'best.pt']:
                old_checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {old_checkpoint}")
    
    def check_early_stopping(self, current_metric: float) -> bool:
        """Check if training should stop early."""
        config = self.early_stopping_config
        
        if not config.enabled:
            return False
        
        should_stop = False
        
        if config.mode == 'min':
            improvement = self.best_metric - current_metric >= config.min_delta
        else:
            improvement = current_metric - self.best_metric >= config.min_delta
        
        if improvement:
            self.best_metric = current_metric
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            logger.info(f"No improvement. Patience: {self.patience_counter}/{config.patience}")
        
        if self.patience_counter >= config.patience:
            logger.info(f"Early stopping triggered after {config.patience} checks without improvement")
            should_stop = True
        
        return should_stop
    
    def get_gpu_stats(self) -> Tuple[List[float], List[float]]:
        """Get GPU utilization and memory statistics."""
        gpu_usage = []
        gpu_memory = []
        
        try:
            import torch
            
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    # Memory allocated
                    mem_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
                    mem_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)  # GB
                    total_mem = torch.cuda.get_device_properties(i).total_mem / (1024 ** 3)
                    
                    gpu_memory.append((mem_allocated / total_mem) * 100)
                
                # Try nvidia-smi for utilization
                try:
                    import subprocess
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,nounits,noheader'],
                        capture_output=True, text=True
                    )
                    gpu_usage = [float(x.strip()) for x in result.stdout.strip().split('\n') if x.strip()]
                except Exception:
                    gpu_usage = [50.0] * len(gpu_memory)  # Fallback
                    
        except Exception as e:
            logger.debug(f"Could not get GPU stats: {e}")
        
        return gpu_usage, gpu_memory
    
    def calculate_eta(self, elapsed_time: float, progress: float) -> str:
        """Calculate estimated time remaining."""
        if progress <= 0:
            return "N/A"
        
        total_estimated = elapsed_time / progress
        remaining = total_estimated - elapsed_time
        
        return str(timedelta(seconds=int(remaining)))
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop.
        
        Returns:
            Dictionary with training results
        """
        import torch
        
        self.start_time = time.time()
        
        # Setup
        logger.info("Setting up training...")
        self.prepare_data()
        self.build_model()
        
        # Get hyperparameters
        hp = self.config.get('hyperparameters', {})
        self.total_epochs = hp.get('epochs', 10)
        
        # Setup components
        self.setup_distributed()
        self.setup_mixed_precision()
        self.setup_gradient_checkpointing()
        
        # Create optimizer and scheduler
        self.optimizer = self.get_optimizer(self.model.parameters())
        total_steps = self.total_epochs * len(self.train_loader)
        self.scheduler = self.get_lr_scheduler(self.optimizer, total_steps)
        
        # Resume from checkpoint if specified
        resume_from = self.config.get('resume_from_checkpoint')
        if resume_from:
            self.load_checkpoint(resume_from)
        
        logger.info(f"Starting training for {self.total_epochs} epochs...")
        logger.info(f"Total estimated steps: {total_steps}")
        
        # Main training loop
        for epoch in range(self.current_epoch + 1, self.total_epochs + 1):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch}/{self.total_epochs}")
            logger.info(f"{'='*60}\n")
            
            # Train one epoch
            train_results = self.train_epoch(epoch)
            
            # Validate
            val_results = self.validate(epoch)
            
            # Calculate epoch duration
            epoch_duration = time.time() - epoch_start_time
            
            # Build metrics object
            metrics = TrainingMetrics(
                epoch=epoch,
                step=self.global_step,
                total_steps=total_steps,
                loss=train_results.get('loss', 0),
                learning_rate=self.optimizer.param_groups[0]['lr'],
                train_loss=train_results.get('loss'),
                val_loss=val_results.get('val_loss'),
                accuracy=val_results.get('accuracy'),
                perplexity=val_results.get('perplexity'),
                throughput=train_results.get('throughput'),
                elapsed_time=time.time() - self.start_time,
            )
            
            # Get GPU stats
            gpu_usage, gpu_memory = self.get_gpu_stats()
            metrics.gpu_usage = gpu_usage
            metrics.gpu_memory = gpu_memory
            metrics.eta = self.calculate_eta(metrics.elapsed_time, self.global_step / total_steps)
            
            # Log metrics
            self.log_metrics(metrics, self.global_step)
            
            # Check for best model
            monitor_metric = val_results.get(self.early_stopping_config.monitor, 0)
            is_best = False
            if self.early_stopping_config.mode == 'min':
                is_best = monitor_metric < self.best_metric
            else:
                is_best = monitor_metric > self.best_metric
            
            # Save checkpoint
            should_save = (
                epoch % self.checkpoint_config.save_every_n_epochs == 0 or
                self.global_step % self.checkpoint_config.save_every_n_steps < len(self.train_loader)
            )
            
            if should_save or is_best:
                self.save_checkpoint(epoch, is_best=is_best)
            
            # Check early stopping
            if self.check_early_stopping(monitor_metric):
                logger.info("Early stopping triggered!")
                break
        
        # Final results
        total_training_time = time.time() - self.start_time
        
        results = {
            'status': 'completed',
            'total_epochs': self.current_epoch,
            'total_steps': self.global_step,
            'final_train_loss': self.metrics_history[-1].train_loss if self.metrics_history else None,
            'final_val_loss': self.metrics_history[-1].val_loss if self.metrics_history else None,
            'best_metric': self.best_metric,
            'training_time_seconds': total_training_time,
            'training_time_human': str(timedelta(seconds=int(total_training_time))),
            'output_directory': str(self.output_dir),
            'checkpoints_saved': len(list((self.output_dir / 'checkpoints').glob('*.pt'))),
        }
        
        # Close logging
        if self.tb_writer:
            self.tb_writer.close()
        if self.wandb_run:
            self.wandb_run.finish()
        
        return results
    
    @abstractmethod
    def export(self, export_config: Dict[str, Any]):
        """Export trained model in various formats."""
        pass


def create_trainer(config: Dict[str, Any]) -> BaseTrainer:
    """Factory function to create appropriate trainer instance."""
    model_type = config.get('model_type', 'language')
    
    trainers = {
        'language': 'training.models.language_model.LanguageModelTrainer',
        'speech-recognition': 'training.models.speech_recognition.SpeechRecognitionTrainer',
        'speech-synthesis': 'training.models.speech_synthesis.SpeechSynthesisTrainer',
        'emotion-recognition': 'training.models.emotion_recognition.EmotionRecognitionTrainer',
        'vision-encoder': 'training.models.vision_encoder.VisionEncoderTrainer',
        'video-generator': 'training.models.video_generator.VideoGeneratorTrainer',
        'motion-generator': 'training.models.motion_generator.MotionGeneratorTrainer',
        'lip-sync': 'training.models.lip_sync.LipSyncTrainer',
        'gesture-model': 'training.models.gesture_model.GestureModelTrainer',
        'memory-module': 'training.models.memory_module.MemoryModuleTrainer',
        'reward-model': 'training.models.reward_model.RewardModelTrainer',
    }
    
    module_path = trainers.get(model_type)
    if not module_path:
        raise ValueError(f"Unknown model type: {model_type}")
    
    parts = module_path.split('.')
    module_name = '.'.join(parts[:-1])
    class_name = parts[-1]
    
    import importlib
    module = importlib.import_module(module_name)
    TrainerClass = getattr(module, class_name)
    
    return TrainerClass(config)

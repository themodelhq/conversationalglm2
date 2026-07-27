#!/usr/bin/env python3
"""
GLM Platform - Main Training Entry Point
========================================

This is the main entry point for training GLM models.
It provides a unified interface for training different model types.

Usage:
    python train.py --config configs/language_model.yaml
    python train.py --model-type language --dataset data/corpus.jsonl
    python train.py --resume checkpoints/latest
"""

import argparse
import os
import sys
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('GLM-Training')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='GLM Platform Training Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train language model with config file
  python train.py --config configs/glm4_language.yaml
  
  # Train vision model
  python train.py --model-type vision --dataset data/vision_data/
  
  # Resume from checkpoint
  python train.py --resume output/checkpoint-1000
  
  # Multi-GPU distributed training
  python train.py --config config.yaml --gpus 4 --distributed deepspeed
        """
    )
    
    # Configuration options
    parser.add_argument('--config', '-c', type=str, 
                        help='Path to YAML configuration file')
    parser.add_argument('--model-type', '-m', type=str,
                        choices=['language', 'speech-recognition', 'speech-synthesis',
                                'emotion-recognition', 'vision-encoder', 'video-generator',
                                'motion-generator', 'lip-sync', 'gesture-model',
                                'memory-module', 'reward-model'],
                        help='Type of model to train')
    
    # Data options
    parser.add_argument('--dataset', '-d', type=str,
                        help='Path to dataset or dataset name')
    parser.add_argument('--validation-split', type=float, default=0.1,
                        help='Validation set split ratio (default: 0.1)')
    
    # Training options
    parser.add_argument('--epochs', '-e', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', '-b', type=int, default=None,
                        help='Training batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=None,
                        help='Peak learning rate')
    parser.add_argument('--warmup-steps', type=int, default=None,
                        help='Number of warmup steps')
    
    # Hardware options
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use (default: 1)')
    parser.add_argument('--gpu-type', type=str, default='auto',
                        help='GPU type (e.g., A100, V100, RTX4090)')
    parser.add_argument('--distributed', type=str, default='ddp',
                        choices=['ddp', 'deepspeed', 'fsdp'],
                        help='Distributed training backend')
    parser.add_argument('--deepspeed-stage', type=int, default=2,
                        help='DeepSpeed optimization stage (1-3)')
    
    # Checkpoint/Resume options
    parser.add_argument('--resume', '-r', type=str, default=None,
                        help='Path to checkpoint for resuming training')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='Output directory for model and logs')
    
    # Export options
    parser.add_argument('--export-formats', nargs='+',
                        choices=['pytorch', 'onnx', 'tensorrt', 'gguf', 'safetensors'],
                        help='Formats to export after training')
    parser.add_argument('--quantize', type=str, default=None,
                        choices=['fp32', 'fp16', 'int8', 'int4', 'awq', 'gptq'],
                        help='Quantization method for export')
    
    # Utility options
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate configuration without training')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r') as f:
        if path.suffix in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif path.suffix == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")


def merge_args_with_config(args, config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge CLI arguments with configuration file."""
    merged = config.copy()
    
    # Override config with CLI args where specified
    if args.model_type:
        merged['model_type'] = args.model_type
    if args.dataset:
        merged['dataset'] = {'path': args.dataset}
    if args.epochs is not None:
        merged.setdefault('hyperparameters', {})['epochs'] = args.epochs
    if args.batch_size is not None:
        merged.setdefault('hyperparameters', {})['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        merged.setdefault('hyperparameters', {})['learning_rate'] = args.learning_rate
    if args.warmup_steps is not None:
        merged.setdefault('hyperparameters', {})['warmup_steps'] = args.warmup_steps
    
    # Hardware settings
    merged.setdefault('hardware', {})
    merged['hardware']['gpus'] = args.gpus
    if args.gpu_type != 'auto':
        merged['hardware']['gpu_type'] = args.gpu_type
    if args.distributed != 'ddp':
        merged['hardware']['distributed_backend'] = args.distributed
    if args.distributed == 'deepspeed':
        merged['hardware']['deepspeed_stage'] = args.deepspeed_stage
    
    # Output directory
    if args.output_dir:
        merged['output_dir'] = args.output_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        merged['output_dir'] = f"output/{merged.get('model_type', 'unknown')}_{timestamp}"
    
    # Export settings
    if args.export_formats:
        merged.setdefault('export', {})['formats'] = args.export_formats
    if args.quantize:
        merged.setdefault('export', {}).setdefault('quantization', {})['method'] = args.quantize
    
    # Resume training
    if args.resume:
        merged['resume_from_checkpoint'] = args.resume
    
    # Seed
    merged['seed'] = args.seed
    
    return merged


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate training configuration."""
    required_fields = ['model_type']
    
    for field in required_fields:
        if field not in config:
            logger.error(f"Missing required field: {field}")
            return False
    
    valid_model_types = [
        'language', 'speech-recognition', 'speech-synthesis',
        'emotion-recognition', 'vision-encoder', 'video-generator',
        'motion-generator', 'lip-sync', 'gesture-model',
        'memory-module', 'reward-model'
    ]
    
    if config['model_type'] not in valid_model_types:
        logger.error(f"Invalid model type: {config['model_type']}")
        logger.info(f"Valid types: {valid_model_types}")
        return False
    
    # Validate hyperparameters if present
    if 'hyperparameters' in config:
        hp = config['hyperparameters']
        if 'learning_rate' in hp and hp['learning_rate'] <= 0:
            logger.error("Learning rate must be positive")
            return False
        if 'batch_size' in hp and hp['batch_size'] <= 0:
            logger.error("Batch size must be positive")
            return False
        if 'epochs' in hp and hp['epochs'] <= 0:
            logger.error("Epochs must be positive")
            return False
    
    logger.info("Configuration validated successfully")
    return True


def setup_training_environment(config: Dict[str, Any]):
    """Setup training environment based on configuration."""
    import torch
    
    # Set random seeds for reproducibility
    seed = config.get('seed', 42)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Configure CUDA
    if torch.cuda.is_available():
        gpus = config.get('hardware', {}).get('gpus', 1)
        if gpus > torch.cuda.device_count():
            logger.warning(f"Requested {gpus} GPUs but only {torch.cuda.device_count()} available")
        
        # Set CUDA memory allocation strategy
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(gpus))
        
        # Enable cuDNN benchmark for performance
        torch.backends.cudnn.benchmark = True
        
        # Enable TF32 for Ampere+ GPUs
        if torch.cuda.is_bf16_supported():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    # Create output directories
    output_dir = Path(config.get('output_dir', 'output'))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'checkpoints').mkdir(exist_ok=True)
    (output_dir / 'logs').mkdir(exist_ok=True)
    (output_dir / 'exports').mkdir(exist_ok=True)
    
    # Save configuration
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    logger.info(f"Output directory: {output_dir.absolute()}")
    return output_dir


def get_trainer_class(model_type: str):
    """Get the appropriate trainer class for the model type."""
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
        raise ValueError(f"No trainer available for model type: {model_type}")
    
    # Dynamic import
    parts = module_path.split('.')
    module_name = '.'.join(parts[:-1])
    class_name = parts[-1]
    
    import importlib
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def main():
    """Main training entry point."""
    args = parse_args()
    
    # Set debug level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=" * 60)
    logger.info("GLM Platform Training Pipeline v2.0")
    logger.info("=" * 60)
    
    try:
        # Load configuration
        if args.config:
            config = load_config(args.config)
            logger.info(f"Loaded config from: {args.config}")
        else:
            # Build config from CLI arguments
            config = {
                'model_type': args.model_type or 'language',
                'dataset': {'path': args.dataset} if args.dataset else {},
                'hyperparameters': {
                    k: v for k, v in [
                        ('epochs', args.epochs),
                        ('batch_size', args.batch_size),
                        ('learning_rate', args.learning_rate),
                        ('warmup_steps', args.warmup_steps),
                    ] if v is not None
                },
            }
        
        # Merge with CLI arguments
        config = merge_args_with_config(args, config)
        
        # Dry run mode
        if args.dry_run:
            logger.info("DRY RUN MODE - Validating configuration only")
            if validate_config(config):
                logger.info("Configuration is valid!")
                print(json.dumps(config, indent=2, default=str))
                return 0
            else:
                logger.error("Configuration validation failed!")
                return 1
        
        # Validate configuration
        if not validate_config(config):
            return 1
        
        # Setup environment
        output_dir = setup_training_environment(config)
        
        # Get trainer class
        TrainerClass = get_trainer_class(config['model_type'])
        
        # Initialize trainer
        logger.info(f"Initializing {config['model_type']} trainer...")
        trainer = TrainerClass(config)
        
        # Start training
        logger.info("Starting training...")
        results = trainer.train()
        
        # Export models if configured
        if config.get('export', {}).get('formats'):
            logger.info("Exporting trained models...")
            trainer.export(config['export'])
        
        logger.info("=" * 60)
        logger.info("Training completed successfully!")
        logger.info(f"Results: {results}")
        logger.info(f"Outputs saved to: {output_dir}")
        logger.info("=" * 60)
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Training failed: {str(e)}")
        return 1


if __name__ == '__main__':
    exit(main())

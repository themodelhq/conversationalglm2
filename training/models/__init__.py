"""
GLM Platform - Training Models Package
=======================================

This package contains all model trainers for the GLM Platform.
"""

from .base_trainer import BaseTrainer, TrainingMetrics, create_trainer
from .language_model import LanguageModelTrainer, GLMDataset, LanguageModelConfig
from .vision_encoder import VisionEncoderTrainer, VisionDataset
from .speech_recognition import SpeechRecognitionTrainer, AudioDataset
from .video_generator import VideoGeneratorTrainer, VideoDataset
from .additional_trainers import (
    EmotionRecognitionTrainer,
    SpeechSynthesisTrainer,
    LipSyncTrainer,
    GestureModelTrainer,
    MemoryModuleTrainer,
    RewardModelTrainer,
)

__all__ = [
    'BaseTrainer',
    'TrainingMetrics',
    'create_trainer',
    'LanguageModelTrainer',
    'VisionEncoderTrainer',
    'SpeechRecognitionTrainer',
    'VideoGeneratorTrainer',
    'EmotionRecognitionTrainer',
    'SpeechSynthesisTrainer',
    'LipSyncTrainer',
    'GestureModelTrainer',
    'MemoryModuleTrainer',
    'RewardModelTrainer',
]

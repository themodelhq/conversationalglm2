# Training Guide

This comprehensive guide covers training GLM models from scratch or fine-tuning existing models.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Training Configuration](#training-configuration)
3. [Model Types](#model-types)
4. [Distributed Training](#distributed-training)
5. [Mixed Precision Training](#mixed-precision-training)
6. [Monitoring & Logging](#monitoring--logging)
7. [Resuming Training](#resuming-training)
8. [Exporting Models](#exporting-models)

---

## Quick Start

### Basic Language Model Training

```bash
cd training

# Train with default settings
python train.py --model-type language --dataset data/my_corpus.jsonl

# Train with configuration file
python train.py --config configs/glm4_chat.yaml
```

### Fine-tune Existing Model

```bash
python train.py \
  --model-type language \
  --config configs/finetune.yaml \
  --resume checkpoints/latest.pt
```

### Multi-GPU Training

```bash
# DDP (default)
python train.py --config config.yaml --gpus 4

# DeepSpeed
python train.py --config config.yaml --gpus 4 --distributed deepspeed --deepspeed-stage 2

# FSDP
python train.py --config config.yaml --gpus 8 --distributed fsdp
```

---

## Training Configuration

### YAML Configuration File

```yaml
# configs/glm4_language.yaml

model_type: language

language_model:
  model_name_or_path: "THUDM/glm-4-9b-chat"
  max_seq_length: 2048
  use_lora: true
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  use_4bit: false          # Set true for QLoRA
  gradient_accumulation_steps: 4

dataset:
  path: "data/training_data.jsonl"
  format: "jsonl"
  validation_split: 0.1
  preprocessing:
    tokenize: true
    normalize: true
    clean: true

hyperparameters:
  learning_rate: 2e-4
  batch_size: 4           # Per GPU
  epochs: 3
  warmup_steps: 100
  weight_decay: 0.01
  optimizer: "adamw"
  scheduler: "cosine"
  mixed_precision: true
  gradient_checkpointing: true

hardware:
  gpus: 4
  gpu_type: "A100"
  distributed_backend: "ddp"

checkpoint:
  save_every_n_steps: 500
  save_every_n_epochs: 1
  keep_last_n: 5

early_stopping:
  enabled: true
  monitor: "val_loss"
  patience: 5
  min_delta: 0.001

logging:
  log_every_n_steps: 10
  tensorboard: true
  wandb: false
  wandb_project: "glm-training"

export:
  formats:
    - pytorch
    - safetensors
    - onnx
  quantization:
    enabled: true
    method: fp16
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | string | - | Path to YAML config file |
| `--model-type` | string | language | Model type to train |
| `--dataset` | string | - | Dataset path |
| `--epochs` | int | 10 | Number of training epochs |
| `--batch-size` | int | 32 | Batch size per GPU |
| `--learning-rate` | float | 1e-4 | Peak learning rate |
| `--gpus` | int | 1 | Number of GPUs |
| `--distributed` | string | ddp | Distributed backend |
| `--resume` | string | - | Checkpoint to resume from |
| `--dry-run` | flag | - | Validate config without training |

---

## Model Types

### 1. Language Model (LLM)

**Use case:** Chat, instruction following, text generation

```yaml
model_type: language

language_model:
  model_name_or_path: "THUDM/glm-4-9b-chat"
  max_seq_length: 4096
  use_lora: true
  lora_r: 16
  lora_target_modules:
    - query_key_value
    - dense
    - dense_h_to_4h
```

**Dataset format (JSONL):**
```jsonl
{"conversations": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi! How can I help you today?"}]}
{"conversations": [{"role": "user", "content": "Explain quantum computing."}, {"role": "assistant", "content": "Quantum computing is..."}]}
```

### 2. Vision Encoder

**Use case:** Image classification, visual understanding

```yaml
model_type: vision-encoder

vision:
  image_size: 224
  num_classes: 1000
  model_name: "google/vit-base-patch16-224"
```

**Dataset structure:**
```
data/images/
├── train/
│   ├── cats/
│   │   ├── image001.jpg
│   │   └── ...
│   └── dogs/
│       └── ...
└── val/
    └── ...
```

### 3. Speech Recognition (ASR)

**Use case:** Audio transcription, speech-to-text

```yaml
model_type: speech-recognition

speech_recognition:
  vocab_size: 5000
  d_model: 144
  num_layers: 16
  sample_rate: 16000
```

**Dataset format:**
```json
{"audio_path": "data/audio/sample.wav", "text": "Hello world", "duration": 2.5}
```

### 4. Video Generator

**Use case:** Text-to-video generation

```yaml
model_type: video-generator

video_generator:
  num_frames: 16
  frame_size: 256
  d_model: 320
  num_layers: 12
```

### 5. Emotion Recognition

**Use case:** Sentiment analysis, affective computing

```yaml
model_type: emotion-recognition

emotion_recognition:
  num_emotions: 7  # neutral, happy, sad, angry, fearful, surprised, disgusted
```

### 6. Other Models

See `training/models/additional_trainers.py` for:
- Speech Synthesis (TTS)
- Lip Sync
- Gesture Generation
- Memory Module
- Reward Model (RLHF)

---

## Distributed Training

### Data Parallelism (DDP)

Default for multi-GPU training:

```bash
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  --master_port=12355 \
  train.py --config config.yaml
```

### DeepSpeed

For large models or memory optimization:

```bash
deepspeed train.py \
  --deepspeed ds_config.json \
  --config config.yaml
```

**DeepSpeed config example (`ds_config.json`):**
```json
{
  "train_batch_size": 32,
  "gradient_accumulation_steps": 4,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 2e-4,
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 2e-4,
      "warmup_num_steps": 100
    }
  },
  "fp16": {"enabled": true},
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {"device": "cpu"},
    "offload_param": {"device": "cpu"}
  }
}
```

### FSDP (Fully Sharded Data Parallel)

For PyTorch 2.x:

```bash
torchrun \
  --nproc_per_node=8 \
  train.py --config config.yaml --distributed fsdp
```

---

## Mixed Precision Training

### Automatic Mixed Precision (AMP)

Enabled by default with `mixed_precision: true`:

```yaml
hyperparameters:
  mixed_precision: true  # Uses FP16 for forward/backward pass
```

### BF16 Training (Ampere+ GPUs)

```yaml
hyperparameters:
  bf16: true  # Better range than FP16
```

### Gradient Checkpointing

Trade compute for memory:

```yaml
hyperparameters:
  gradient_checkpointing: true  # Reduces memory by ~30%
```

---

## Monitoring & Logging

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir output/logs/tensorboard

# View at http://localhost:6006
```

### Weights & Biases

```yaml
logging:
  wandb: true
  wandb_project: "my-glm-training"
  wandb_entity: "my-team"
```

### Console Output

Training progress is logged every N steps:

```
2024-01-15 10:23:45 | INFO | Epoch 1/3 | Step 100/2000 | Loss: 2.3456 | LR: 1.98e-04 | Val Loss: 2.1234 | ETA: 2h 34m
```

---

## Resuming Training

### From Last Checkpoint

```bash
python train.py --config config.yaml --resume output/checkpoints/latest.pt
```

### From Specific Checkpoint

```bash
python train.py --config config.yaml --resume output/checkpoints/checkpoint-epoch-2-step-1500.pt
```

### Checkpoint Contents

Checkpoints include:
- Model weights
- Optimizer state
- Scheduler state
- Scaler state (for AMP)
- Training metrics history
- Configuration used

---

## Exporting Models

### During Training

Configure exports in your training config:

```yaml
export:
  formats:
    - pytorch
    - safetensors
    - onnx
    - gguf
  quantization:
    method: fp16
```

### Post-Training Export

```bash
# Use the export API
curl -X POST http://localhost:3000/api/export \
  -H "Content-Type: application/json" \
  -d '{
    "modelName": "glm-4-chat",
    "formats": ["pytorch", "onnx", "gguf"],
    "quantization": { "method": "int4" }
  }'
```

### Export Formats Comparison

| Format | Size (9B model) | Speed | Best For |
|--------|-----------------|-------|----------|
| FP32 PyTorch | ~18 GB | Baseline | Training |
| FP16 SafeTensors | ~9 GB | 1.2x | Production |
| INT4 GGUF | ~2.5 GB | 0.8x | Local inference |
| ONNX | ~9 GB | 1.5x | Cross-platform |
| TensorRT | ~7 GB | 3x | NVIDIA GPU |

---

## Tips & Best Practices

### Memory Optimization

1. **Enable gradient checkpointing** for large models
2. **Use LoRA/QLoRA** for fine-tuning instead of full finetuning
3. **Reduce batch size** and increase gradient accumulation
4. **Use DeepSpeed ZeRO-2/3** for very large models

### Speed Optimization

1. **Use compiled model** (PyTorch 2.x): `model = torch.compile(model)`
2. **Increase dataloader workers**: `num_workers=8`
3. **Pin memory**: `pin_memory=True`
4. **Use TF32** on Ampere+: `torch.backends.cuda.matmul.allow_tf32 = True`

### Quality Optimization

1. **Start with lower learning rate**: 1e-5 to 2e-4
2. **Use warmup**: 5-10% of total steps
3. **Monitor validation loss** for overfitting
4. **Use early stopping** with patience of 3-5 checks

"""
GLM Platform - Language Model Trainer
=====================================

This module provides training functionality for GLM language models,
including:
- Causal language modeling
- Instruction tuning
- Chat fine-tuning
- RLHF (Reinforcement Learning from Human Feedback)
"""

import os
import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from base_trainer import BaseTrainer, TrainingMetrics

logger = logging.getLogger('GLM-Training.Language')


@dataclass
class LanguageModelConfig:
    """Configuration specific to language model training."""
    model_name_or_path: str = 'THUDM/glm-4-9b-chat'
    max_seq_length: int = 2048
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None
    use_4bit: bool = False
    use_8bit: bool = False
    gradient_accumulation_steps: int = 4
    packing: bool = True  # Pack short sequences together


class GLMDataset(Dataset):
    """Custom dataset for GLM language model training."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        mode: str = 'train',
        instruction_format: str = 'chat',
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.instruction_format = instruction_format
        
        # Load data
        import json
        self.data = []
        
        path = Path(data_path)
        if path.is_file():
            with open(path, 'r') as f:
                if path.suffix == '.jsonl':
                    for line in f:
                        line = line.strip()
                        if line:
                            self.data.append(json.loads(line))
                else:
                    self.data = json.load(f)
        elif path.is_dir():
            # Load all JSONL files in directory
            for file in path.glob('**/*.jsonl'):
                with open(file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.data.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def format_instruction(self, sample: Dict[str, Any]) -> str:
        """Format sample according to instruction template."""
        if self.instruction_format == 'chat':
            messages = []
            if 'system' in sample and sample['system']:
                messages.append({'role': 'system', 'content': sample['system']})
            
            if 'conversations' in sample:
                messages.extend(sample['conversations'])
            else:
                messages.append({'role': 'user', 'content': sample.get('input', sample.get('question', ''))})
                if 'output' in sample or 'answer' in sample:
                    messages.append({'role': 'assistant', 'content': sample.get('output', sample.get('answer', ''))})
            
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        elif self.instruction_format == 'alpaca':
            return (
                f"Below is an instruction that describes a task. "
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{sample.get('instruction', '')}\n\n"
                f"### Input:\n{sample.get('input', '')}\n\n"
                f"### Response:\n{sample.get('output', sample.get('response', ''))}"
            )
        
        else:
            # Plain text
            return sample.get('text', '')
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        text = self.format_instruction(sample)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # For causal LM, labels are same as input_ids (shifted during loss computation)
        labels = input_ids.clone()
        
        # Mask padding tokens in labels
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


class LanguageModelTrainer(BaseTrainer):
    """Trainer for GLM language models."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Parse language model config
        lm_config = config.get('language_model', {})
        self.lm_config = LanguageModelConfig(**lm_config)
        
        # Initialize tokenizer
        self.tokenizer = None
        self.model = None
        self.train_loader = None
        self.val_loader = None
    
    def prepare_data(self):
        """Prepare training and validation datasets."""
        logger.info("Preparing language model data...")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.lm_config.model_name_or_path,
            trust_remote_code=True,
            padding_side='right',  # For efficient generation
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        dataset_config = self.config.get('dataset', {})
        data_path = dataset_config.get('path', './data')
        val_split = dataset_config.get('validation_split', 0.1)
        
        # Create full dataset
        full_dataset = GLMDataset(
            data_path=data_path,
            tokenizer=self.tokenizer,
            max_length=self.lm_config.max_seq_length,
            instruction_format=self.config.get('instruction_format', 'chat'),
        )
        
        # Split into train/val
        total_size = len(full_dataset)
        val_size = int(total_size * val_split)
        train_size = total_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.get('seed', 42)),
        )
        
        # Create data loaders
        hp = self.config.get('hyperparameters', {})
        batch_size = hp.get('batch_size', 4)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=default_collate_fn,
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=default_collate_fn,
        )
        
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    def build_model(self):
        """Build the language model."""
        logger.info("Building language model...")
        
        # Quantization config for 4-bit/8-bit
        quantization_config = None
        if self.lm_config.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
            )
        elif self.lm_config.use_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.lm_config.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if not quantization_config else None,
            quantization_config=quantization_config,
            device_map='auto' if self.num_gpus > 1 else None,
        )
        
        # Setup LoRA if enabled
        if self.lm_config.use_lora:
            logger.info(f"Setting up LoRA (r={self.lm_config.lora_r}, alpha={self.lm_config.lora_alpha})")
            
            if self.lm_config.use_4bit or self.lm_config.use_8bit:
                self.model = prepare_model_for_kbit_training(self.model)
            
            target_modules = self.lm_config.lora_target_modules or [
                'query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h'
            ]
            
            lora_config = LoraConfig(
                r=self.lm_config.lora_r,
                lora_alpha=self.lm_config.lora_alpha,
                lora_dropout=self.lm_config.lora_dropout,
                target_modules=target_modules,
                task_type=TaskType.CAUSAL_LM,
                bias='none',
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        # Move to device
        if not next(self.model.parameters()).is_cuda and self.device.type == 'cuda':
            self.model = self.model.to(self.device)
        
        # Gradient checkpointing
        if self.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        epoch_start_time = time.time()
        
        hp = self.config.get('hyperparameters', {})
        grad_accum_steps = self.lm_config.gradient_accumulation_steps
        
        for step, batch in enumerate(self.train_loader):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass with mixed precision
            if self.use_mixed_precision and self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss / grad_accum_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (step + 1) % grad_accum_steps == 0:
                    # Clip gradients
                    if hasattr(self, 'scaler'):
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / grad_accum_steps
                
                loss.backward()
                
                if (step + 1) % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
            
            total_loss += loss.item() * grad_accum_steps
            num_batches += 1
            self.global_step += 1
            
            # Log progress
            if step % self.logging_config.log_every_n_steps == 0:
                elapsed = time.time() - epoch_start_time
                throughput = (step + 1) * hp.get('batch_size', 4) / elapsed
                
                metrics = TrainingMetrics(
                    epoch=epoch,
                    step=self.global_step,
                    total_steps=len(self.train_loader) * self.total_epochs,
                    loss=loss.item() * grad_accum_steps,
                    learning_rate=self.optimizer.param_groups[0]['lr'],
                    throughput=throughput,
                )
                self.log_metrics(metrics, self.global_step)
        
        avg_loss = total_loss / max(num_batches, 1)
        epoch_duration = time.time() - epoch_start_time
        
        return {
            'loss': avg_loss,
            'duration': epoch_duration,
            'batches': num_batches,
        }
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                
                total_loss += outputs.loss.item()
                num_batches += 1
                
                # Calculate perplexity
                logits = outputs.logits
                predictions = logits.argmax(dim=-1)
                
                # Count correct predictions (excluding padding)
                mask = labels != -100
                if mask.any():
                    correct = (predictions[mask] == labels[mask]).float().sum().item()
                    total_correct += correct
                    total_tokens += mask.sum().item()
        
        avg_loss = total_loss / max(num_batches, 1)
        accuracy = total_correct / max(total_tokens, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            'val_loss': avg_loss,
            'accuracy': accuracy,
            'perplexity': perplexity,
        }
    
    def export(self, export_config: Dict[str, Any]):
        """Export trained model in various formats."""
        logger.info("Exporting language model...")
        
        export_dir = self.output_dir / 'exports'
        export_dir.mkdir(parents=True, exist_ok=True)
        
        formats = export_config.get('formats', ['pytorch'])
        quantization = export_config.get('quantization', {})
        quant_method = quantization.get('method', 'fp16')
        
        # Merge LoRA weights if using LoRA
        if self.lm_config.use_lora:
            logger.info("Merging LoRA weights...")
            from peft import PeftModel
            if isinstance(self.model, PeftModel):
                self.model = self.model.merge_and_unload()
        
        for fmt in formats:
            if fmt == 'pytorch' or fmt == 'safetensors':
                self._export_pytorch(export_dir, fmt, quant_method)
            elif fmt == 'onnx':
                self._export_onnx(export_dir, quant_method)
            elif fmt == 'gguf':
                self._export_gguf(export_dir, quant_method)
            elif fmt == 'tensorrt':
                self._export_tensorrt(export_dir, quant_method)
        
        logger.info(f"Export complete. Files saved to: {export_dir}")
    
    def _export_pytorch(self, export_dir: Path, fmt: str, quant_method: str):
        """Export to PyTorch or SafeTensors format."""
        suffix = '.safetensors' if fmt == 'safetensors' else '.pt'
        
        # Save model
        model_path = export_dir / f'model{suffix}'
        
        if fmt == 'safetensors':
            try:
                from safetensors.torch import save_file
                state_dict = self.model.state_dict()
                save_file(state_dict, str(model_path))
            except ImportError:
                # Fallback to PyTorch
                torch.save(self.model.state_dict(), export_dir / 'model.pt')
        else:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.model.config,
            }, model_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(str(export_dir / 'tokenizer'))
        
        logger.info(f"Saved {fmt} model to {model_path}")
    
    def _export_onnx(self, export_dir: Path, quant_method: str):
        """Export to ONNX format."""
        try:
            import torch.onnx
            
            onnx_path = export_dir / 'model.onnx'
            
            # Prepare dummy input
            dummy_input = torch.randint(
                0, self.tokenizer.vocab_size,
                (1, 128), device=self.device
            )
            dummy_attention = torch.ones(1, 128, device=self.device)
            
            # Export
            torch.onnx.export(
                self.model,
                (dummy_input, dummy_attention),
                str(onnx_path),
                input_names=['input_ids', 'attention_mask'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch', 1: 'sequence'},
                    'attention_mask': {0: 'batch', 1: 'sequence'},
                    'logits': {0: 'batch', 1: 'sequence'},
                },
                opset_version=17,
            )
            
            logger.info(f"Saved ONNX model to {onnx_path}")
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
    
    def _export_gguf(self, export_dir: Path, quant_method: str):
        """Export to GGUF format (for llama.cpp)."""
        try:
            # This requires the ctransformers or llama.cpp conversion tools
            logger.info("GGUF export requires external conversion tools")
            logger.info("Please use convert_hf_to_gguf.py from llama.cpp")
            
            # Save intermediate HF format first
            temp_dir = export_dir / 'hf_intermediate'
            temp_dir.mkdir(exist_ok=True)
            self.model.save_pretrained(str(temp_dir))
            self.tokenizer.save_pretrained(str(temp_dir))
            
        except Exception as e:
            logger.error(f"GGUF export failed: {e}")
    
    def _export_tensorrt(self, export_dir: Path, quant_method: str):
        """Export to TensorRT format."""
        try:
            import tensorrt as trt
            logger.info("TensorRT export requires TensorRT installation")
            logger.info("Please ensure TensorRT is properly installed")
        except ImportError:
            logger.error("TensorRT not installed")


def default_collate_fn(batch):
    """Default collate function for batching."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }

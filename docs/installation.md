# Installation Guide

This guide covers setting up GLM Platform for development and production.

## System Requirements

### Minimum Requirements (Development)

| Component | Specification |
|-----------|----------------|
| CPU | 4 cores |
| RAM | 8 GB |
| Storage | 20 GB SSD |
| GPU | Not required for development |
| OS | Ubuntu 20.04+, macOS 12+, Windows 10+ |

### Recommended Requirements (Training)

| Component | Specification |
|-----------|----------------|
| CPU | 16+ cores |
| RAM | 64+ GB |
| Storage | 100+ GB NVMe SSD |
| GPU | NVIDIA A100/H100 (80GB VRAM) or RTX 4090 (24GB) |
| CUDA | 11.8+ / cuDNN 8+ |
| Driver | 525+ |

## Prerequisites

### Node.js & Bun

```bash
# Install Bun (recommended)
curl -fsSL https://bun.sh/install | bash

# Or install Node.js via nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 20
nvm use 20
```

### Python (for Training)

```bash
# Install Python 3.10+
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# .\venv\Scripts\activate  # Windows
```

### Docker (Optional)

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
sudo apt install docker-compose-plugin
```

## Development Setup

### 1. Clone Repository

```bash
git clone https://github.com/glm-platform/glm-platform.git
cd glm-platform
```

### 2. Install Frontend Dependencies

```bash
bun install
```

### 3. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit with your configuration
nano .env
```

Required environment variables:

```env
# Application
NODE_ENV=development
PORT=3000

# Database (SQLite for dev)
DATABASE_URL="file:./dev.db"

# AI SDK (optional, for local testing)
ZAI_API_KEY=your-api-key-here
```

### 4. Initialize Database

```bash
bun run db:push
```

### 5. Start Development Server

```bash
bun run dev
```

The application will be available at `http://localhost:3000`

## Production Setup

### Option 1: Docker Deployment

```bash
# Build and run with Docker Compose
cd deployment/docker
docker compose up -d --build
```

### Option 2: Kubernetes Deployment

```bash
kubectl apply -f deployment/kubernetes/
```

### Option 3: Netlify/Render

See deployment-specific guides:
- [Netlify Guide](../deployment/netlify/)
- [Render Guide](../deployment/render/)

## Training Environment Setup

### 1. Install NVIDIA Drivers & CUDA

```bash
# Check GPU status
nvidia-smi

# Verify CUDA installation
nvcc --version
```

### 2. Install PyTorch with CUDA

```bash
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install Training Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:

```
# Core ML
transformers>=4.36.0
accelerate>=0.25.0
peft>=0.7.0
bitsandbytes>=0.41.0
deepspeed>=0.12.0

# Vision
torchvision>=0.16.0
Pillow>=10.0.0

# Audio
torchaudio>=2.1.0

# Utilities
tensorboard>=2.15.0
wandb>=0.16.0
pyyaml>=6.0
tqdm>=4.66.0
```

### 4. Verify Installation

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
```

## Troubleshooting

### Port Already in Use

```bash
# Find process using port 3000
lsof -i :3000

# Kill the process
kill -9 <PID>
```

### Memory Issues

```bash
# Clear bun/node cache
bun pm cache rm
npm cache clean --force
```

### GPU Not Detected

```bash
# Check NVIDIA driver status
nvidia-smi

# Reinstall drivers if needed
sudo apt reinstall nvidia-driver-535
```

### Database Errors

```bash
# Reset database
bun run db:reset

# Regenerate Prisma client
bun run db:generate
```

## Next Steps

After installation, check out:

- [Training Guide](training.md) - Start training models
- [Dataset Guide](datasets.md) - Prepare your data
- [API Reference](api.md) - Build integrations

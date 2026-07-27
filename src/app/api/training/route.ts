import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db';
import type { TrainingConfig, TrainingStatus, TrainingMetrics, ModelType } from '@/types';

// In-memory training store (in production, use database)
const trainingStore = new Map<string, {
  config: TrainingConfig;
  status: TrainingStatus;
  metrics: TrainingMetrics[];
  startTime?: Date;
  endTime?: Date;
}>();

// Simulated training progress
const trainingProgress = new Map<string, NodeJS.Timeout>();

export async function GET() {
  try {
    // Return all training jobs
    const trainings = Array.from(trainingStore.entries()).map(([id, data]) => ({
      id,
      ...data.config,
      status: data.status,
      metrics: data.metrics.slice(-100), // Last 100 metrics
      startTime: data.startTime,
      endTime: data.endTime,
    }));

    return NextResponse.json({ trainings });
  } catch (error) {
    console.error('Error fetching trainings:', error);
    return NextResponse.json(
      { error: 'Failed to fetch training jobs' },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action, trainingId, config } = body;

    switch (action) {
      case 'create':
        return await createTraining(config);
      
      case 'start':
        return await startTraining(trainingId);
      
      case 'pause':
        return await pauseTraining(trainingId);
      
      case 'resume':
        return await resumeTraining(trainingId);
      
      case 'stop':
        return await stopTraining(trainingId);
      
      case 'delete':
        return await deleteTraining(trainingId);
      
      default:
        return NextResponse.json({ error: 'Invalid action' }, { status: 400 });
    }
  } catch (error: any) {
    console.error('Training API Error:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to process request' },
      { status: 500 }
    );
  }
}

async function createTraining(config: Partial<TrainingConfig>) {
  const id = `train_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  
  const newConfig: TrainingConfig = {
    id,
    name: config.name || 'New Training Job',
    modelType: config.modelType || 'language',
    dataset: config.dataset || {
      name: 'default-dataset',
      path: '/data/datasets/default',
      format: 'jsonl',
      size: 0,
      validationSplit: 0.1,
      preprocessing: {
        tokenize: true,
        normalize: true,
        augment: false,
        clean: true,
        shard: false,
        balance: false,
      },
    },
    hyperparameters: config.hyperparameters || {
      learningRate: 1e-4,
      batchSize: 32,
      epochs: 10,
      warmupSteps: 100,
      weightDecay: 0.01,
      gradientAccumulation: 1,
      mixedPrecision: true,
      gradientCheckpointing: true,
      optimizer: 'adamw',
      scheduler: 'cosine',
    },
    hardware: config.hardware || {
      gpus: 1,
      gpuType: 'A100',
      multiNode: false,
      nodes: 1,
      distributedBackend: 'ddp',
    },
    export: config.export || {
      formats: ['pytorch', 'safetensors'],
      quantization: {
        enabled: false,
        method: 'fp16',
        bits: 16,
        groupSize: 128,
      },
      optimization: {
        fuseLayers: true,
        optimizeFor: 'throughput',
        targetDevice: 'cuda',
      },
    },
    status: 'idle',
    createdAt: new Date(),
  };

  trainingStore.set(id, {
    config: newConfig,
    status: 'idle',
    metrics: [],
  });

  return NextResponse.json({
    success: true,
    training: newConfig,
    message: 'Training job created successfully',
  });
}

async function startTraining(trainingId: string) {
  const training = trainingStore.get(trainingId);
  
  if (!training) {
    return NextResponse.json({ error: 'Training job not found' }, { status: 404 });
  }

  if (training.status === 'training') {
    return NextResponse.json({ error: 'Training already in progress' }, { status: 400 });
  }

  // Update status
  training.status = 'preparing';
  training.startTime = new Date();

  // Simulate training preparation
  setTimeout(() => {
    training.status = 'training';
    simulateTrainingProgress(trainingId);
  }, 2000);

  return NextResponse.json({
    success: true,
    message: 'Training started',
    status: training.status,
  });
}

function simulateTrainingProgress(trainingId: string) {
  const training = trainingStore.get(trainingId);
  if (!training) return;

  const totalSteps = training.config.hyperparameters.epochs * 2000; // Simulated steps per epoch
  let currentStep = 0;
  let currentEpoch = 1;
  let loss = Math.random() * 2 + 1; // Starting loss

  const interval = setInterval(() => {
    const trainingData = trainingStore.get(trainingId);
    if (!trainingData || trainingData.status !== 'training') {
      clearInterval(interval);
      return;
    }

    currentStep++;
    
    // Update every 50 steps (simulating batch processing)
    if (currentStep % 50 === 0) {
      // Decrease loss over time with some noise
      loss = Math.max(0.01, loss * (0.995 + Math.random() * 0.01));
      
      // Update epoch
      if (currentStep % 2000 === 0) {
        currentEpoch++;
      }

      const metric: TrainingMetrics = {
        epoch: currentEpoch,
        step: currentStep,
        totalSteps,
        loss,
        learningRate: training.config.hyperparameters.learningRate * Math.pow(0.999, currentStep / 50),
        trainLoss: loss * (1 + Math.random() * 0.05),
        valLoss: loss * (1 + Math.random() * 0.1),
        accuracy: Math.min(0.99, 0.5 + (currentStep / totalSteps) * 0.45 + Math.random() * 0.05),
        perplexity: Math.exp(loss),
        gpuUsage: Array.from({ length: training.config.hardware.gpus }, () => 80 + Math.random() * 20),
        gpuMemory: Array.from({ length: training.config.hardware.gpus }, () => 60 + Math.random() * 30),
        throughput: 2000 + Math.random() * 1000,
        elapsed_time: currentStep * 2,
        eta: `${Math.floor((totalSteps - currentStep) * 2 / 3600)}h ${Math.floor(((totalSteps - currentStep) * 2 % 3600) / 60)}m`,
      };

      trainingData.metrics.push(metric);

      // Check if training is complete
      if (currentStep >= totalSteps) {
        clearInterval(interval);
        trainingData.status = 'completed';
        trainingData.endTime = new Date();
      }
    }
  }, 100); // Update every 100ms

  trainingProgress.set(trainingId, interval);
}

async function pauseTraining(trainingId: string) {
  const training = trainingStore.get(trainingId);
  
  if (!training) {
    return NextResponse.json({ error: 'Training job not found' }, { status: 404 });
  }

  if (training.status !== 'training') {
    return NextResponse.json({ error: 'Training is not running' }, { status: 400 });
  }

  // Clear progress simulation
  const interval = trainingProgress.get(trainingId);
  if (interval) {
    clearInterval(interval);
    trainingProgress.delete(trainingId);
  }

  training.status = 'paused';

  return NextResponse.json({
    success: true,
    message: 'Training paused',
    status: training.status,
  });
}

async function resumeTraining(trainingId: string) {
  const training = trainingStore.get(trainingId);
  
  if (!training) {
    return NextResponse.json({ error: 'Training job not found' }, { status: 404 });
  }

  if (training.status !== 'paused') {
    return NextResponse.json({ error: 'Training is not paused' }, { status: 400 });
  }

  training.status = 'training';
  simulateTrainingProgress(trainingId);

  return NextResponse.json({
    success: true,
    message: 'Training resumed',
    status: training.status,
  });
}

async function stopTraining(trainingId: string) {
  const training = trainingStore.get(trainingId);
  
  if (!training) {
    return NextResponse.json({ error: 'Training job not found' }, { status: 404 });
  }

  // Clear progress simulation
  const interval = trainingProgress.get(trainingId);
  if (interval) {
    clearInterval(interval);
    trainingProgress.delete(trainingId);
  }

  training.status = 'idle';
  training.endTime = new Date();

  return NextResponse.json({
    success: true,
    message: 'Training stopped',
    status: training.status,
  });
}

async function deleteTraining(trainingId: string) {
  const training = trainingStore.get(trainingId);
  
  if (!training) {
    return NextResponse.json({ error: 'Training job not found' }, { status: 404 });
  }

  // Stop training if running
  if (training.status === 'training' || training.status === 'paused') {
    const interval = trainingProgress.get(trainingId);
    if (interval) {
      clearInterval(interval);
      trainingProgress.delete(trainingId);
    }
  }

  trainingStore.delete(trainingId);

  return NextResponse.json({
    success: true,
    message: 'Training job deleted',
  });
}

// Get training metrics endpoint
export async function PUT(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const trainingId = searchParams.get('id');
  
  if (!trainingId) {
    return NextResponse.json({ error: 'Training ID required' }, { status: 400 });
  }

  const training = trainingStore.get(trainingId);
  
  if (!training) {
    return NextResponse.json({ error: 'Training job not found' }, { status: 404 });
  }

  return NextResponse.json({
    metrics: training.metrics,
    status: training.status,
    currentStep: training.metrics.length > 0 ? training.metrics[training.metrics.length - 1].step : 0,
  });
}

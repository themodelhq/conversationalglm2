import { NextRequest, NextResponse } from 'next/server';
import type { TrainingJob, ModelType, TrainingConfig } from '@/types';

// In-memory storage for training jobs (in production, use a database)
const trainingJobs = new Map<string, TrainingJob>();

export async function GET() {
  try {
    const jobs = Array.from(trainingJobs.values());
    
    return NextResponse.json({
      success: true,
      data: { jobs },
    });
  } catch (error: any) {
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action, jobId, config } = body;

    switch (action) {
      case 'create': {
        // Create a new training job
        if (!config?.modelName || !config?.modelType) {
          return NextResponse.json(
            { success: false, error: 'Model name and type are required' },
            { status: 400 }
          );
        }

        const newJob: TrainingJob = {
          id: `training-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
          name: config.modelName,
          modelType: config.modelType as ModelType,
          status: 'pending',
          progress: 0,
          epochs: config.epochs || 10,
          currentEpoch: 0,
          config: config as TrainingConfig,
          startedAt: new Date(),
        };

        trainingJobs.set(newJob.id, newJob);

        // Simulate starting training after a short delay
        setTimeout(() => {
          newJob.status = 'running';
          simulateTrainingProgress(newJob.id);
        }, 1000);

        return NextResponse.json({
          success: true,
          data: { job: newJob },
        });
      }

      case 'cancel': {
        if (!jobId) {
          return NextResponse.json(
            { success: false, error: 'Job ID is required' },
            { status: 400 }
          );
        }

        const job = trainingJobs.get(jobId);
        if (!job) {
          return NextResponse.json(
            { success: false, error: 'Job not found' },
            { status: 404 }
          );
        }

        job.status = 'cancelled';
        trainingJobs.set(jobId, job);

        return NextResponse.json({
          success: true,
          data: { job },
        });
      }

      case 'list': {
        const jobs = Array.from(trainingJobs.values());
        return NextResponse.json({
          success: true,
          data: { jobs },
        });
      }

      default:
        return NextResponse.json(
          { success: false, error: `Unknown action: ${action}` },
          { status: 400 }
        );
    }
  } catch (error: any) {
    console.error('Training API Error:', error);
    return NextResponse.json(
      { success: false, error: error.message || 'Failed to process request' },
      { status: 500 }
    );
  }
}

export async function DELETE(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const jobId = searchParams.get('jobId');

    if (!jobId) {
      return NextResponse.json(
        { success: false, error: 'Job ID is required' },
        { status: 400 }
      );
    }

    const deleted = trainingJobs.delete(jobId);
    
    if (!deleted) {
      return NextResponse.json(
        { success: false, error: 'Job not found' },
        { status: 404 }
      );
    }

    return NextResponse.json({
      success: true,
      message: 'Training job deleted',
    });
  } catch (error: any) {
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 500 }
    );
  }
}

// Simulate training progress
function simulateTrainingProgress(jobId: string) {
  const interval = setInterval(() => {
    const job = trainingJobs.get(jobId);
    if (!job || job.status !== 'running') {
      clearInterval(interval);
      return;
    }

    const increment = Math.random() * 10 + 2;
    job.progress = Math.min(job.progress + increment, 100);
    job.currentEpoch = Math.min(
      Math.floor((job.progress / 100) * job.epochs),
      job.epochs
    );

    trainingJobs.set(jobId, { ...job });

    if (job.progress >= 100) {
      job.status = 'completed';
      job.completedAt = new Date();
      trainingJobs.set(jobId, { ...job });
      clearInterval(interval);
    }
  }, 2000);
}

import { NextRequest, NextResponse } from 'next/server';
import type { ExportJob, ExportFormat } from '@/types';

// In-memory storage for export jobs (in production, use a database)
const exportJobs = new Map<string, ExportJob>();

export async function GET() {
  try {
    const jobs = Array.from(exportJobs.values());
    
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
    const { action, jobId, modelName, format } = body;

    switch (action) {
      case 'create': {
        // Create a new export job
        if (!modelName || !format) {
          return NextResponse.json(
            { success: false, error: 'Model name and format are required' },
            { status: 400 }
          );
        }

        const validFormats: ExportFormat[] = ['pytorch', 'onnx', 'tensorrt', 'gguf', 'safetensors'];
        if (!validFormats.includes(format)) {
          return NextResponse.json(
            { 
              success: false, 
              error: `Invalid format. Valid formats: ${validFormats.join(', ')}` 
            },
            { status: 400 }
          );
        }

        const newJob: ExportJob = {
          id: `export-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
          modelName,
          format,
          status: 'pending',
          progress: 0,
          createdAt: new Date(),
        };

        exportJobs.set(newJob.id, newJob);

        // Simulate starting export after a short delay
        setTimeout(() => {
          newJob.status = 'running';
          simulateExportProgress(newJob.id);
        }, 500);

        return NextResponse.json({
          success: true,
          data: { job: newJob },
        });
      }

      case 'list': {
        const jobs = Array.from(exportJobs.values());
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
    console.error('Export API Error:', error);
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

    const deleted = exportJobs.delete(jobId);
    
    if (!deleted) {
      return NextResponse.json(
        { success: false, error: 'Job not found' },
        { status: 404 }
      );
    }

    return NextResponse.json({
      success: true,
      message: 'Export job deleted',
    });
  } catch (error: any) {
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 500 }
    );
  }
}

// Simulate export progress
function simulateExportProgress(jobId: string) {
  const interval = setInterval(() => {
    const job = exportJobs.get(jobId);
    if (!job || job.status !== 'running') {
      clearInterval(interval);
      return;
    }

    const increment = Math.random() * 15 + 5;
    job.progress = Math.min(job.progress + increment, 100);

    // Generate output path based on format
    const extensions: Record<ExportFormat, string> = {
      pytorch: '.pt',
      onnx: '.onnx',
      tensorrt: '.trt',
      gguf: '.gguf',
      safetensors: '.safetensors',
    };

    if (job.progress >= 100) {
      job.status = 'completed';
      job.outputPath = `/exports/${job.modelName}${extensions[job.format]}`;
      exportJobs.set(jobId, { ...job });
      clearInterval(interval);
    } else {
      exportJobs.set(jobId, { ...job });
    }
  }, 800);
}

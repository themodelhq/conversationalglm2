import { NextRequest, NextResponse } from 'next/server';
import type { ModelExport, ExportFormat, QuantizationType } from '@/types';

// In-memory export store
const exportStore = new Map<string, ModelExport>();

// Export configurations for different formats
const exportConfigs: Record<ExportFormat, {
  extension: string;
  mimeType: string;
  description: string;
}> = {
  pytorch: { extension: '.pt', mimeType: 'application/octet-stream', description: 'PyTorch model file' },
  onnx: { extension: '.onnx', mimeType: 'application/octet-stream', description: 'ONNX model file' },
  tensorrt: { extension: '.engine', mimeType: 'application/octet-stream', description: 'TensorRT engine file' },
  gguf: { extension: '.gguf', mimeType: 'application/octet-stream', description: 'GGUF quantized model' },
  safetensors: { extension: '.safetensors', mimeType: 'application/octet-stream', description: 'SafeTensors model' },
};

export async function GET() {
  try {
    const exports = Array.from(exportStore.values());
    return NextResponse.json({ exports });
  } catch (error) {
    console.error('Error fetching exports:', error);
    return NextResponse.json(
      { error: 'Failed to fetch exports' },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { modelName, formats, quantization, optimization } = body;

    if (!modelName || !formats || formats.length === 0) {
      return NextResponse.json(
        { error: 'modelName and formats are required' },
        { status: 400 }
      );
    }

    const createdExports: ModelExport[] = [];

    // Create export jobs for each format
    for (const format of formats as ExportFormat[]) {
      const id = `exp_${Date.now()}_${format}`;
      const config = exportConfigs[format];
      
      const exp: ModelExport = {
        id,
        modelName,
        format,
        path: `/exports/${modelName}/${modelName}${config.extension}`,
        size: 0,
        quantization: (quantization?.method || 'fp16') as QuantizationType,
        status: 'exporting',
        createdAt: new Date(),
      };

      exportStore.set(id, exp);
      createdExports.push(exp);

      // Simulate export process
      simulateExport(id);
    }

    return NextResponse.json({
      success: true,
      exports: createdExports,
      message: `Created ${createdExports.length} export job(s)`,
    });

  } catch (error: any) {
    console.error('Export API Error:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to create export job' },
      { status: 500 }
    );
  }
}

async function simulateExport(exportId: string) {
  const exp = exportStore.get(exportId);
  if (!exp) return;

  // Simulate export time based on format and quantization
  const baseTime = 2000; // Base time in ms
  const formatMultiplier: Record<ExportFormat, number> = {
    pytorch: 1,
    onnx: 1.5,
    tensorrt: 3,
    gguf: 2.5,
    safetensors: 0.8,
  };
  
  const quantMultiplier: Record<string, number> = {
    fp32: 0.5,
    fp16: 1,
    int8: 1.5,
    int4: 2.5,
    awq: 3,
    gptq: 3,
  };

  const exportTime = baseTime * (formatMultiplier[exp.format] || 1) * (quantMultiplier[exp.quantization] || 1);

  setTimeout(() => {
    const updatedExp = exportStore.get(exportId);
    if (!updatedExp) return;

    // Calculate simulated metrics
    const baseSize = getBaseModelSize(updatedExp.modelName);
    const compressionRatio = getCompressionRatio(exp.quantization);
    
    updatedExp.status = 'completed';
    updatedExp.size = Math.floor(baseSize / compressionRatio);
    updatedExp.metrics = {
      latency_ms: getSimulatedLatency(exp.format),
      throughput_samples: Math.floor(Math.random() * 1000) + 100,
      memory_mb: Math.floor(baseSize / compressionRatio / (1024 * 1024)),
      compression_ratio: compressionRatio,
      accuracy_delta: exp.quantization === 'int4' ? -0.02 : 
                      exp.quantization === 'int8' ? -0.005 : 0,
    };

    exportStore.set(exportId, updatedExp);
  }, exportTime);
}

function getBaseModelSize(modelName: string): number {
  // Return approximate model sizes in bytes
  const sizes: Record<string, number> = {
    'glm-4-base': 18 * 1024 * 1024 * 1024, // ~18GB for 9B params FP16
    'glm-4-chat': 18 * 1024 * 1024 * 1024,
    'glm-4v': 20 * 1024 * 1024 * 1024,     // Vision model slightly larger
    'glm-4-speech': 5 * 1024 * 1024 * 1024,
    'glm-4-tts': 3 * 1024 * 1024 * 1024,
    'vision-encoder-v2': 2 * 1024 * 1024 * 1024,
    'video-gen-base': 15 * 1024 * 1024 * 1024,
  };
  
  return sizes[modelName] || 10 * 1024 * 1024 * 1024; // Default 10GB
}

function getCompressionRatio(quantization: string): number {
  switch (quantization) {
    case 'fp32': return 1;
    case 'fp16': return 2;
    case 'int8': return 4;
    case 'int4': return 7.5;
    case 'awq': return 6;
    case 'gptq': return 6;
    default: return 2;
  }
}

function getSimulatedLatency(format: ExportFormat): number {
  switch (format) {
    case 'pytorch': return 45 + Math.floor(Math.random() * 30);
    case 'onnx': return 35 + Math.floor(Math.random() * 25);
    case 'tensorrt': return 15 + Math.floor(Math.random() * 10);
    case 'gguf': return 60 + Math.floor(Math.random() * 40);
    case 'safetensors': return 45 + Math.floor(Math.random() * 30);
    default: return 50;
  }
}

// Delete an export
export async function DELETE(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const exportId = searchParams.get('id');
  
  if (!exportId) {
    return NextResponse.json({ error: 'Export ID required' }, { status: 400 });
  }

  const deleted = exportStore.delete(exportId);
  
  if (!deleted) {
    return NextResponse.json({ error: 'Export not found' }, { status: 404 });
  }

  return NextResponse.json({
    success: true,
    message: 'Export deleted',
  });
}

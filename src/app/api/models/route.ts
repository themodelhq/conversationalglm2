import { NextRequest, NextResponse } from 'next/server';

// Model registry - in production, this would come from a database or model hub
const modelRegistry = [
  {
    id: 'glm-4-base',
    name: 'GLM-4 Base',
    description: 'Base language model with 9B parameters',
    type: 'language',
    size: '18GB (FP16)',
    params: '9B',
    status: 'available',
    capabilities: ['text-generation', 'conversation', 'reasoning'],
    versions: ['v1.0', 'v1.1'],
    latestVersion: 'v1.1',
  },
  {
    id: 'glm-4-chat',
    name: 'GLM-4 Chat',
    description: 'Chat-optimized model with instruction following',
    type: 'language',
    size: '18GB (FP16)',
    params: '9B',
    status: 'available',
    capabilities: ['chat', 'instruction-following', 'function-calling'],
    versions: ['v1.0', 'v1.2'],
    latestVersion: 'v1.2',
  },
  {
    id: 'glm-4-plus',
    name: 'GLM-4 Plus',
    description: 'Enhanced model with improved reasoning',
    type: 'language',
    size: '36GB (FP16)',
    params: '18B',
    status: 'available',
    capabilities: ['chat', 'reasoning', 'code-generation', 'analysis'],
    versions: ['v1.0'],
    latestVersion: 'v1.0',
  },
  {
    id: 'glm-4v',
    name: 'GLM-4 Vision',
    description: 'Multimodal model with vision understanding',
    type: 'vision',
    size: '20GB (FP16)',
    params: '9B',
    status: 'available',
    capabilities: ['vision', 'image-understanding', 'ocr', 'visual-reasoning'],
    versions: ['v1.0', 'v1.1'],
    latestVersion: 'v1.1',
  },
  {
    id: 'glm-4-speech',
    name: 'GLM-4 Speech ASR',
    description: 'Automatic speech recognition model',
    type: 'speech-recognition',
    size: '5GB (FP16)',
    params: '2B',
    status: 'available',
    capabilities: ['asr', 'transcription', 'speaker-diarization'],
    versions: ['v1.0'],
    latestVersion: 'v1.0',
  },
  {
    id: 'glm-4-tts',
    name: 'GLM-4 TTS',
    description: 'Text-to-speech synthesis model',
    type: 'speech-synthesis',
    size: '3GB (FP16)',
    params: '1.5B',
    status: 'beta',
    capabilities: ['tts', 'voice-synthesis', 'emotion-control'],
    versions: ['v0.9-beta'],
    latestVersion: 'v0.9-beta',
  },
  {
    id: 'vision-encoder-v2',
    name: 'Vision Encoder V2',
    description: 'High-performance vision encoder for image understanding',
    type: 'vision-encoder',
    size: '2GB (FP16)',
    params: '800M',
    status: 'available',
    capabilities: ['feature-extraction', 'image-classification', 'embedding'],
    versions: ['v2.0', 'v2.1'],
    latestVersion: 'v2.1',
  },
  {
    id: 'video-gen-base',
    name: 'Video Generator Base',
    description: 'Text-to-video generation model',
    type: 'video-generator',
    size: '15GB (FP16)',
    params: '7B',
    status: 'beta',
    capabilities: ['video-generation', 'text-to-video', 'animation'],
    versions: ['v0.5-beta'],
    latestVersion: 'v0.5-beta',
  },
  {
    id: 'lip-sync-model',
    name: 'Lip Sync Model',
    description: 'Audio-driven lip synchronization model',
    type: 'lip-sync',
    size: '500MB (FP16)',
    params: '200M',
    status: 'beta',
    capabilities: ['lip-sync', 'talking-head', 'facial-animation'],
    versions: ['v1.0-alpha'],
    latestVersion: 'v1.0-alpha',
  },
  {
    id: 'emotion-model',
    name: 'Emotion Recognition Model',
    description: 'Speech and text emotion recognition',
    type: 'emotion-recognition',
    size: '200MB (FP16)',
    params: '100M',
    status: 'available',
    capabilities: ['emotion-recognition', 'sentiment-analysis', 'affect-computing'],
    versions: ['v1.0', 'v1.1'],
    latestVersion: 'v1.1',
  },
];

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const type = searchParams.get('type');
    const search = searchParams.get('search');
    
    let filteredModels = [...modelRegistry];
    
    // Filter by type
    if (type && type !== 'all') {
      filteredModels = filteredModels.filter(m => m.type === type);
    }
    
    // Filter by search
    if (search) {
      const query = search.toLowerCase();
      filteredModels = filteredModels.filter(m => 
        m.name.toLowerCase().includes(query) ||
        m.description.toLowerCase().includes(query) ||
        m.id.includes(query)
      );
    }

    return NextResponse.json({
      success: true,
      models: filteredModels,
      total: filteredModels.length,
      types: [...new Set(modelRegistry.map(m => m.type))],
    });

  } catch (error: any) {
    console.error('Models API Error:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to fetch models' },
      { status: 500 }
    );
  }
}

// Get specific model details
export async function PUT(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const modelId = searchParams.get('id');
    
    if (!modelId) {
      return NextResponse.json({ error: 'Model ID required' }, { status: 400 });
    }

    const model = modelRegistry.find(m => m.id === modelId);
    
    if (!model) {
      return NextResponse.json({ error: 'Model not found' }, { status: 404 });
    }

    // Return detailed model information
    return NextResponse.json({
      success: true,
      model: {
        ...model,
        details: {
          architecture: getArchitecture(model.type),
          trainingData: getTrainingData(model.id),
          benchmarks: getBenchmarks(model.id),
          license: 'Apache 2.0',
          languages: ['en', 'zh', 'es', 'fr', 'de', 'ja', 'ko', 'pt', 'ru', 'ar'],
          contextWindow: getContextWindow(model.type),
          maxOutputTokens: getMaxOutputTokens(model.type),
        },
        exportFormats: ['pytorch', 'onnx', 'gguf', 'safetensors'],
        quantizationOptions: ['fp32', 'fp16', 'int8', 'int4'],
      },
    });

  } catch (error: any) {
    console.error('Model Details Error:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to fetch model details' },
      { status: 500 }
    );
  }
}

function getArchitecture(type: string): string {
  const architectures: Record<string, string> = {
    language: 'Transformer Decoder (GLM Architecture)',
    vision: 'Vision Transformer + Language Model',
    'speech-recognition': 'Conformer Encoder-Decoder',
    'speech-synthesis': 'VITS-based Autoregressive',
    'vision-encoder': 'ViT-L/14 with CLIP-style training',
    'video-generator': 'Diffusion Transformer (DiT)',
    'lip-sync': 'Motion-aware Temporal Convolutional Network',
    'emotion-recognition': 'Multi-modal Transformer',
  };
  return architectures[type] || 'Custom Architecture';
}

function getTrainingData(modelId: string): string {
  const dataSources: Record<string, string> = {
    'glm-4-base': 'Web corpus, books, code repositories (~3T tokens)',
    'glm-4-chat': 'Instruction datasets, dialogues, RLHF data',
    'glm-4-plus': 'Extended pre-training data, reasoning chains',
    'glm-4v': 'Image-text pairs, visual question answering datasets',
    'glm-4-speech': 'Audio corpora, transcriptions (100K+ hours)',
    'glm-4-tts': 'Speech recordings, emotion-labeled audio',
  };
  return dataSources[modelId] || 'Proprietary training dataset';
}

function getBenchmarks(modelId: string): Record<string, number> {
  const benchmarks: Record<string, Record<string, number>> = {
    'glm-4-base': { MMLU: 72.5, HellaSwag: 85.3, WinoGrande: 82.1 },
    'glm-4-chat': { 'MT-Bench': 8.2, AlpacaEval: 89.5, HumanEval: 45.2 },
    'glm-4v': { VQAv2: 78.3, GQA: 62.1, TextVQA: 58.9 },
    'glm-4-speech': { 'WER-LibriSpeech': 2.8, 'WER-CommonVoice': 5.2 },
  };
  return benchmarks[modelId] || {};
}

function getContextWindow(type: string): number {
  const windows: Record<string, number> = {
    language: 128000,
    vision: 32000,
    'speech-recognition': 30000, // ~30 seconds
    'speech-synthesis': 4096,
    default: 8192,
  };
  return windows[type] || windows.default;
}

function getMaxOutputTokens(type: string): number {
  const limits: Record<string, number> = {
    language: 16384,
    vision: 4096,
    default: 2048,
  };
  return limits[type] || limits.default;
}

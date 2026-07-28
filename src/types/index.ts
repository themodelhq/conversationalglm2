// GLM Platform Type Definitions

// Chat types
export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  attachments?: Attachment[];
  metadata?: MessageMetadata;
}

export interface Attachment {
  type: 'image' | 'file';
  url: string;
  name: string;
  size?: number;
}

export interface MessageMetadata {
  model?: string;
  tokens?: number;
  latency?: number;
  emotion?: EmotionData;
}

export interface EmotionData {
  primary: string;
  confidence: number;
}

// Chat session
export interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
  settings: ChatSettings;
}

export interface ChatSettings {
  model: string;
  temperature: number;
  maxTokens: number;
  systemPrompt: string;
}

// Model types
export interface AIModel {
  id: string;
  name: string;
  type: ModelType;
  provider: string;
  status: 'available' | 'training' | 'exported';
  capabilities: string[];
}

export type ModelType = 
  | 'llm'
  | 'vision'
  | 'asr'
  | 'tts'
  | 'image'
  | 'video'
  | 'emotion'
  | 'multimodal';

// Training types
export interface TrainingJob {
  id: string;
  name: string;
  modelType: ModelType;
  status: TrainingStatus;
  progress: number;
  epochs: number;
  currentEpoch: number;
  config: TrainingConfig;
  startedAt?: Date;
  completedAt?: Date;
  outputDir?: string;
}

export type TrainingStatus = 
  | 'pending'
  | 'running'
  | 'paused'
  | 'completed'
  | 'failed'
  | 'cancelled';

export interface TrainingConfig {
  learningRate: number;
  batchSize: number;
  epochs: number;
  optimizer: string;
  mixedPrecision: boolean;
  distributedTraining?: DistributedConfig;
}

export interface DistributedConfig {
  enabled: boolean;
  strategy: 'ddp' | 'deepspeed' | 'fsdp';
  numGpus: number;
}

// Export types
export interface ExportJob {
  id: string;
  modelName: string;
  format: ExportFormat;
  status: ExportStatus;
  progress: number;
  outputPath?: string;
  createdAt: Date;
}

export type ExportFormat = 'pytorch' | 'onnx' | 'tensorrt' | 'gguf' | 'safetensors';
export type ExportStatus = 'pending' | 'running' | 'completed' | 'failed';

// Image generation
export interface ImageGenerationRequest {
  prompt: string;
  size: string;
  count: number;
}

export interface ImageGenerationResult {
  success: boolean;
  url: string;
  filename: string;
  size: number;
  dimensions: string;
}

// API Response types
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  metadata?: Record<string, unknown>;
}

// Settings
export interface AppSettings {
  theme: 'light' | 'dark' | 'system';
  sidebarOpen: boolean;
  activeView: ViewType;
}

export type ViewType = 
  | 'chat'
  | 'training'
  | 'models'
  | 'export'
  | 'settings';

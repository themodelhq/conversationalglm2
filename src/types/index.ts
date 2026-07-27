// Multimodal AI Platform Types

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  attachments?: Attachment[];
  metadata?: MessageMetadata;
}

export interface Attachment {
  id: string;
  type: 'image' | 'audio' | 'video' | 'file';
  url: string;
  name: string;
  size?: number;
}

export interface MessageMetadata {
  model?: string;
  tokens?: number;
  latency?: number;
  emotion?: EmotionData;
  functions?: FunctionCall[];
}

export interface EmotionData {
  primary: string;
  confidence: number;
  secondary?: string;
  arousal?: number;
  valence?: number;
}

export interface FunctionCall {
  name: string;
  arguments: Record<string, unknown>;
  result?: unknown;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
  model: string;
  settings: SessionSettings;
}

export interface SessionSettings {
  temperature: number;
  maxTokens: number;
  systemPrompt: string;
  enableMemory: boolean;
  enableRAG: boolean;
  enableVision: boolean;
  enableVoice: boolean;
  emotionMode: 'neutral' | 'expressive' | 'dramatic';
}

export interface TrainingConfig {
  id: string;
  name: string;
  modelType: ModelType;
  dataset: DatasetConfig;
  hyperparameters: Hyperparameters;
  hardware: HardwareConfig;
  export: ExportConfig;
  status: TrainingStatus;
  createdAt: Date;
}

export type ModelType = 
  | 'language'
  | 'speech-recognition'
  | 'speech-synthesis'
  | 'emotion-recognition'
  | 'vision-encoder'
  | 'video-generator'
  | 'motion-generator'
  | 'lip-sync'
  | 'gesture-model'
  | 'memory-module'
  | 'reward-model';

export type TrainingStatus = 
  | 'idle'
  | 'preparing'
  | 'training'
  | 'validating'
  | 'exporting'
  | 'completed'
  | 'failed'
  | 'paused';

export interface DatasetConfig {
  name: string;
  path: string;
  format: string;
  size: number;
  validationSplit: number;
  preprocessing: PreprocessingConfig;
}

export interface PreprocessingConfig {
  tokenize: boolean;
  normalize: boolean;
  augment: boolean;
  clean: boolean;
  shard: boolean;
  balance: boolean;
}

export interface Hyperparameters {
  learningRate: number;
  batchSize: number;
  epochs: number;
  warmupSteps: number;
  weightDecay: number;
  gradientAccumulation: number;
  mixedPrecision: boolean;
  gradientCheckpointing: boolean;
  optimizer: OptimizerType;
  scheduler: SchedulerType;
}

export type OptimizerType = 'adam' | 'adamw' | 'sgd' | 'adamw8bit';
export type SchedulerType = 'cosine' | 'linear' | 'constant' | 'warmup-cosine';

export interface HardwareConfig {
  gpus: number;
  gpuType: string;
  multiNode: boolean;
  nodes: number;
  distributedBackend: 'ddp' | 'deepspeed' | 'fsdp';
  deepSpeedStage?: number;
}

export interface ExportConfig {
  formats: ExportFormat[];
  quantization: QuantizationConfig;
  optimization: OptimizationConfig;
}

export type ExportFormat = 'pytorch' | 'onnx' | 'tensorrt' | 'gguf' | 'safetensors';
export type QuantizationType = 'fp32' | 'fp16' | 'int8' | 'int4' | 'awq' | 'gptq';

export interface QuantizationConfig {
  enabled: boolean;
  method: QuantizationType;
  bits: number;
  groupSize: number;
}

export interface OptimizationConfig {
  fuseLayers: boolean;
  optimizeFor: 'latency' | 'throughput' | 'memory';
  targetDevice: 'cpu' | 'cuda' | 'tensorrt';
}

export interface TrainingMetrics {
  epoch: number;
  step: number;
  totalSteps: number;
  loss: number;
  learningRate: number;
  trainLoss?: number;
  valLoss?: number;
  accuracy?: number;
  perplexity?: number;
  gpuUsage?: number[];
  gpuMemory?: number[];
  gpuTemperature?: number[];
  throughput?: number;
  elapsed_time?: number;
  eta?: string;
}

export interface ModelExport {
  id: string;
  modelName: string;
  format: ExportFormat;
  path: string;
  size: number;
  quantization: QuantizationType;
  status: 'exporting' | 'completed' | 'failed';
  createdAt: Date;
  metrics?: ExportMetrics;
}

export interface ExportMetrics {
  latency_ms?: number;
  throughput_samples?: number;
  memory_mb?: number;
  compression_ratio?: number;
  accuracy_delta?: number;
}

export interface RAGDocument {
  id: string;
  content: string;
  metadata: Record<string, unknown>;
  embedding?: number[];
  chunkIndex: number;
  source: string;
}

export interface MemoryEntry {
  id: string;
  type: 'fact' | 'preference' | 'event' | 'context';
  content: string;
  importance: number;
  createdAt: Date;
  lastAccessed: Date;
  accessCount: number;
  embedding?: number[];
}

export interface UserProfile {
  id: string;
  name: string;
  preferences: UserPreferences;
  memories: MemoryEntry[];
  conversationHistory: string[];
  createdAt: Date;
}

export interface UserPreferences {
  language: string;
  responseStyle: 'formal' | 'casual' | 'friendly' | 'professional';
  topics: string[];
  avoidTopics: string[];
  notificationSettings: NotificationSettings;
}

export interface NotificationSettings {
  email: boolean;
  push: boolean;
  trainingComplete: boolean;
  errors: boolean;
}

export interface VideoGenerationConfig {
  prompt: string;
  negativePrompt?: string;
  width: number;
  height: number;
  frames: number;
  fps: number;
  guidanceScale: number;
  numInferenceSteps: number;
  seed?: number;
  enableLipSync: boolean;
  enableGesture: boolean;
  emotion?: string;
  referenceImage?: string;
  audioPath?: string;
}

export interface LipSyncConfig {
  audioPath: string;
  videoPath?: string;
  faceImage: string;
  expression: ExpressionType;
  smoothness: number;
}

export type ExpressionType = 'neutral' | 'happy' | 'sad' | 'angry' | 'surprised' | 'fearful' | 'disgusted';

export interface GestureConfig {
  gestureType: GestureType;
  intensity: number;
  duration: number;
  syncWithSpeech: boolean;
}

export type GestureType = 
  | 'wave'
  | 'point'
  | 'thumbs-up'
  | 'nod'
  | 'shake'
  | 'open-palms'
  | 'thinking'
  | 'explaining';

export interface APIKey {
  id: string;
  name: string;
  key: string;
  permissions: string[];
  createdAt: Date;
  lastUsed?: Date;
  usageCount: number;
  rateLimit: number;
}

export interface DashboardStats {
  totalConversations: number;
  totalMessages: number;
  activeTrainings: number;
  completedModels: number;
  gpuHoursUsed: number;
  storageUsed: number;
  apiCallsToday: number;
  activeUsers: number;
}

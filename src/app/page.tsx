'use client'

import React, { useState, useRef, useEffect } from 'react';
import { useAppStore, getApiUrl } from '@/store';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { 
  MessageSquare, 
  Send, 
  Plus, 
  Trash2, 
  Sparkles,
  Image,
  Brain,
  Download,
  Moon,
  Sun,
  Menu,
  X,
  Loader2,
  Play,
  Pause,
  Square,
  Settings,
  Cpu,
  Database,
  FileCode,
  CheckCircle,
  AlertCircle,
  Upload,
  Zap
} from 'lucide-react';

// Model types for training
const MODEL_TYPES = [
  { id: 'llm', name: 'Language Model (LLM)', description: 'Text generation & understanding', icon: Brain },
  { id: 'vision', name: 'Vision Model', description: 'Image classification & analysis', icon: Image },
  { id: 'asr', name: 'Speech Recognition (ASR)', description: 'Audio to text conversion', icon: MessageSquare },
  { id: 'tts', name: 'Text-to-Speech (TTS)', description: 'Text to audio synthesis', icon: Sparkles },
  { id: 'multimodal', name: 'Multimodal Model', description: 'Combined text, image, audio', icon: Zap },
];

// Export formats
const EXPORT_FORMATS = [
  { id: 'pytorch', name: 'PyTorch (.pt)', extension: '.pt' },
  { id: 'onnx', name: 'ONNX (.onnx)', extension: '.onnx' },
  { id: 'safetensors', name: 'SafeTensors (.safetensors)', extension: '.safetensors' },
  { id: 'gguf', name: 'GGUF (.gguf)', extension: '.gguf' },
];

export default function GLMPlatform() {
  const {
    sessions,
    activeSessionId,
    isLoading,
    settings,
    trainingJobs,
    exportJobs,
    createSession,
    deleteSession,
    setActiveSession,
    addMessage,
    updateSessionTitle,
    setLoading,
    toggleSidebar,
    setSettings,
    setActiveView,
    addTrainingJob,
    updateTrainingJob,
    addExportJob,
    updateExportJob,
  } = useAppStore();

  const [inputMessage, setInputMessage] = useState('');
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Training state
  const [trainingConfig, setTrainingConfig] = useState({
    modelName: '',
    modelType: 'llm',
    epochs: 10,
    learningRate: 0.001,
    batchSize: 32,
    optimizer: 'adam',
    mixedPrecision: true,
  });

  // Export state
  const [exportConfig, setExportConfig] = useState({
    modelName: '',
    format: 'pytorch' as const,
  });

  const activeSession = sessions.find(s => s.id === activeSessionId);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [activeSession?.messages]);

  // Create new session if none exists
  useEffect(() => {
    if (sessions.length === 0) {
      createSession();
    }
  }, [sessions.length, createSession]);

  // Handle sending message - uses local API first, then falls back
  const handleSendMessage = async () => {
    if (!inputMessage.trim() || !activeSessionId) return;

    const userMessage = {
      id: `msg-${Date.now()}`,
      role: 'user' as const,
      content: inputMessage.trim(),
      timestamp: new Date(),
    };

    addMessage(activeSessionId, userMessage);
    
    if (activeSession && activeSession.messages.length === 0) {
      updateSessionTitle(activeSessionId, inputMessage.trim().slice(0, 30) + '...');
    }

    setInputMessage('');
    setLoading(true);

    try {
      // Use getApiUrl to determine correct endpoint (local or remote backend)
      const chatUrl = getApiUrl('/api/chat');
      let response = await fetch(chatUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage.content,
          sessionId: activeSessionId,
          settings: activeSession?.settings,
        }),
      });

      // If API call fails, we'll use fallback response
      if (!response.ok) {
        throw new Error(`API responded with status: ${response.status}`);
      }

      const data = await response.json();

      if (data.success && data.data?.content) {
        const assistantMessage = {
          id: `msg-${Date.now()}-resp`,
          role: 'assistant' as const,
          content: data.data.content,
          timestamp: new Date(),
          metadata: data.data.metadata,
        };
        addMessage(activeSessionId, assistantMessage);
      } else {
        throw new Error(data.error || 'No content in response');
      }
    } catch (error: any) {
      console.log('Using fallback mode:', error.message);
      
      // Fallback: Simulate AI response for demo/offline mode
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const fallbackResponses = [
        "I'm GLM, your AI assistant! I can help you with various tasks including:\n\n• **Chat & Conversation** - Ask me anything!\n• **Code Generation** - I write code in any language\n• **Analysis** - Data, text, or image analysis\n• **Creative Writing** - Stories, poems, content\n• **Problem Solving** - Break down complex problems\n\nHow can I assist you today?",
        "That's a great question! Let me help you with that.\n\nBased on my understanding, I'd suggest breaking this down into smaller steps. Would you like me to elaborate on any specific aspect?",
        "I understand what you're looking for. Here's my take:\n\n1. First, let's identify the core requirements\n2. Then we can explore possible solutions\n3. Finally, we'll implement the best approach\n\nShall I proceed with step 1?",
        "Interesting! I'd love to help you with this.\n\nAs a multimodal AI powered by Z.ai SDK, I can assist with text, images, code, and more. What specific aspect would you like to focus on?",
      ];
      
      const randomResponse = fallbackResponses[Math.floor(Math.random() * fallbackResponses.length)];
      
      const assistantMessage = {
        id: `msg-${Date.now()}-resp`,
        role: 'assistant' as const,
        content: randomResponse,
        timestamp: new Date(),
        metadata: {
          model: 'GLM-4 Plus (Demo Mode)',
          tokens: Math.floor(Math.random() * 500) + 100,
          latency: Math.floor(Math.random() * 1000) + 200,
        },
      };
      
      addMessage(activeSessionId, assistantMessage);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const toggleTheme = () => {
    const newTheme = settings.theme === 'dark' ? 'light' : 'dark';
    setSettings({ theme: newTheme });
    document.documentElement.classList.toggle('dark', newTheme === 'dark');
  };

  // Training handlers
  const handleStartTraining = async () => {
    if (!trainingConfig.modelName) return;

    const jobId = `training-${Date.now()}`;
    
    const newJob = {
      id: jobId,
      name: trainingConfig.modelName,
      modelType: trainingConfig.modelType as any,
      status: 'running' as const,
      progress: 0,
      epochs: trainingConfig.epochs,
      currentEpoch: 0,
      config: {
        learningRate: trainingConfig.learningRate,
        batchSize: trainingConfig.batchSize,
        epochs: trainingConfig.epochs,
        optimizer: trainingConfig.optimizer,
        mixedPrecision: trainingConfig.mixedPrecision,
      },
      startedAt: new Date(),
    };

    addTrainingJob(newJob);

    // Simulate training progress
    simulateTrainingProgress(jobId);
  };

  const simulateTrainingProgress = (jobId: string) => {
    const interval = setInterval(() => {
      const job = trainingJobs.find(j => j.id === jobId);
      if (!job || job.status !== 'running') {
        clearInterval(interval);
        return;
      }

      const newProgress = Math.min(job.progress + Math.random() * 15, 100);
      const newEpoch = Math.min(Math.floor(newProgress / (100 / job.epochs)), job.epochs);

      updateTrainingJob(jobId, {
        progress: newProgress,
        currentEpoch: newEpoch,
      });

      if (newProgress >= 100) {
        updateTrainingJob(jobId, {
          status: 'completed',
          completedAt: new Date(),
        });
        clearInterval(interval);
      }
    }, 1500);
  };

  const handleCancelTraining = (jobId: string) => {
    updateTrainingJob(jobId, { status: 'cancelled' });
  };

  // Export handler
  const handleStartExport = async () => {
    if (!exportConfig.modelName) return;

    const jobId = `export-${Date.now()}`;
    
    const newJob = {
      id: jobId,
      modelName: exportConfig.modelName,
      format: exportConfig.format,
      status: 'running' as const,
      progress: 0,
      createdAt: new Date(),
    };

    addExportJob(newJob);

    // Simulate export progress
    const interval = setInterval(() => {
      const job = exportJobs.find(j => j.id === jobId);
      if (!job) {
        clearInterval(interval);
        return;
      }

      const newProgress = Math.min(job.progress + Math.random() * 20, 100);
      updateExportJob(jobId, { progress: newProgress });

      if (newProgress >= 100) {
        updateExportJob(jobId, {
          status: 'completed',
          outputPath: `/exports/${exportConfig.modelName}${EXPORT_FORMATS.find(f => f.id === exportConfig.format)?.extension}`,
        });
        clearInterval(interval);
      }
    }, 800);
  };

  // Render training view
  const renderTrainingView = () => (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* New Training Job */}
      <Card className="glow-purple">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Cpu className="w-5 h-5 text-primary" />
            Start New Training Job
          </CardTitle>
          <CardDescription>
            Configure and start a new model training session
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Model Name */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Model Name</label>
            <Input
              placeholder="my-custom-model"
              value={trainingConfig.modelName}
              onChange={(e) => setTrainingConfig(prev => ({ ...prev, modelName: e.target.value }))}
            />
          </div>

          {/* Model Type */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Model Type</label>
            <Select value={trainingConfig.modelType} onValueChange={(v) => setTrainingConfig(prev => ({ ...prev, modelType: v }))}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {MODEL_TYPES.map(type => (
                  <SelectItem key={type.id} value={type.id}>
                    <span className="flex items-center gap-2">
                      <type.icon className="w-4 h-4" />
                      {type.name}
                    </span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Training Parameters Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Epochs */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Epochs: {trainingConfig.epochs}</label>
              <Slider
                value={[trainingConfig.epochs]}
                onValueChange={([v]) => setTrainingConfig(prev => ({ ...prev, epochs: v }))}
                min={1}
                max={100}
                step={1}
              />
            </div>

            {/* Learning Rate */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Learning Rate: {trainingConfig.learningRate}</label>
              <Slider
                value={[trainingConfig.learningRate]}
                onValueChange={([v]) => setTrainingConfig(prev => ({ ...prev, learningRate: v }))}
                min={0.0001}
                max={0.01}
                step={0.0001}
              />
            </div>

            {/* Batch Size */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Batch Size: {trainingConfig.batchSize}</label>
              <Slider
                value={[trainingConfig.batchSize]}
                onValueChange={([v]) => setTrainingConfig(prev => ({ ...prev, batchSize: v }))}
                min={8}
                max={256}
                step={8}
              />
            </div>

            {/* Optimizer */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Optimizer</label>
              <Select value={trainingConfig.optimizer} onValueChange={(v) => setTrainingConfig(prev => ({ ...prev, optimizer: v }))}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="adam">Adam</SelectItem>
                  <SelectItem value="adamw">AdamW</SelectItem>
                  <SelectItem value="sgd">SGD</SelectItem>
                  <SelectItem value="rmsprop">RMSprop</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* Mixed Precision */}
          <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
            <div>
              <p className="font-medium text-sm">Mixed Precision Training</p>
              <p className="text-xs text-muted-foreground">Use FP16 for faster training on compatible GPUs</p>
            </div>
            <Switch
              checked={trainingConfig.mixedPrecision}
              onCheckedChange={(checked) => setTrainingConfig(prev => ({ ...prev, mixedPrecision: checked }))}
            />
          </div>

          <Button 
            onClick={handleStartTraining} 
            disabled={!trainingConfig.modelName || trainingJobs.some(j => j.status === 'running')}
            className="w-full gap-2"
            size="lg"
          >
            <Play className="w-4 h-4" />
            Start Training
          </Button>
        </CardContent>
      </Card>

      {/* Active Training Jobs */}
      {trainingJobs.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="w-5 h-5" />
              Training Jobs ({trainingJobs.length})
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {trainingJobs.map((job) => (
              <div key={job.id} className="p-4 rounded-lg border border-border space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <Badge variant={
                      job.status === 'completed' ? 'default' :
                      job.status === 'failed' ? 'destructive' :
                      job.status === 'cancelled' ? 'secondary' : 'default'
                    }>
                      {job.status === 'running' && <Loader2 className="w-3 h-3 animate-spin mr-1" />}
                      {job.status === 'completed' && <CheckCircle className="w-3 h-3 mr-1" />}
                      {job.status === 'failed' && <AlertCircle className="w-3 h-3 mr-1" />}
                      {job.status.charAt(0).toUpperCase() + job.status.slice(1)}
                    </Badge>
                    <span className="font-medium">{job.name}</span>
                    <Badge variant="outline">{job.modelType.toUpperCase()}</Badge>
                  </div>
                  
                  {job.status === 'running' && (
                    <Button variant="destructive" size="sm" onClick={() => handleCancelTraining(job.id)}>
                      <Square className="w-3 h-3 mr-1" />
                      Cancel
                    </Button>
                  )}
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Progress</span>
                    <span>{Math.round(job.progress)}% (Epoch {job.currentEpoch}/{job.epochs})</span>
                  </div>
                  <Progress value={job.progress} />
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs text-muted-foreground">
                  <div>LR: {job.config.learningRate}</div>
                  <div>Batch: {job.config.batchSize}</div>
                  <div>Optimizer: {job.config.optimizer}</div>
                  <div>Mixed: {job.config.mixedPrecision ? 'Yes' : 'No'}</div>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {/* Model Types Info */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {MODEL_TYPES.map(({ id, name, description, icon: Icon }) => (
          <Card key={id} className="hover:border-primary/50 transition-colors">
            <CardContent className="p-4">
              <Icon className="w-8 h-8 text-primary mb-2" />
              <h3 className="font-semibold text-sm">{name}</h3>
              <p className="text-xs text-muted-foreground mt-1">{description}</p>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );

  // Render export view
  const renderExportView = () => (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* New Export Job */}
      <Card className="glow-cyan">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Download className="w-5 h-5 text-primary" />
            Export Model
          </CardTitle>
          <CardDescription>
            Export a trained model to various formats
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Model Name */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Model Name</label>
            <Input
              placeholder="my-trained-model"
              value={exportConfig.modelName}
              onChange={(e) => setExportConfig(prev => ({ ...prev, modelName: e.target.value }))}
            />
          </div>

          {/* Export Format */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Export Format</label>
            <Select value={exportConfig.format} onValueChange={(v) => setExportConfig(prev => ({ ...prev, format: v as any }))}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {EXPORT_FORMATS.map(format => (
                  <SelectItem key={format.id} value={format.id}>
                    <span className="flex items-center gap-2">
                      <FileCode className="w-4 h-4" />
                      {format.name}
                    </span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <Button 
            onClick={handleStartExport} 
            disabled={!exportConfig.modelName}
            className="w-full gap-2"
            size="lg"
          >
            <Download className="w-4 h-4" />
            Start Export
          </Button>
        </CardContent>
      </Card>

      {/* Active Export Jobs */}
      {exportJobs.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileCode className="w-5 h-5" />
              Export Jobs ({exportJobs.length})
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {exportJobs.map((job) => (
              <div key={job.id} className="p-4 rounded-lg border border-border space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <Badge variant={job.status === 'completed' ? 'default' : 'secondary'}>
                      {job.status === 'running' && <Loader2 className="w-3 h-3 animate-spin mr-1" />}
                      {job.status === 'completed' && <CheckCircle className="w-3 h-3 mr-1" />}
                      {job.status.charAt(0).toUpperCase() + job.status.slice(1)}
                    </Badge>
                    <span className="font-medium">{job.modelName}</span>
                    <Badge variant="outline">{job.format.toUpperCase()}</Badge>
                  </div>
                  
                  {job.status === 'completed' && job.outputPath && (
                    <Button size="sm" variant="outline">
                      <Download className="w-3 h-3 mr-1" />
                      Download
                    </Button>
                  )}
                </div>

                {job.status === 'running' && (
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Exporting...</span>
                      <span>{Math.round(job.progress)}%</span>
                    </div>
                    <Progress value={job.progress} />
                  </div>
                )}

                {job.status === 'completed' && job.outputPath && (
                  <p className="text-xs text-muted-foreground">
                    Output: {job.outputPath}
                  </p>
                )}
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {/* Format Info */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {EXPORT_FORMATS.map(({ id, name, extension }) => (
          <Card key={id} className="hover:border-primary/50 transition-colors">
            <CardContent className="p-4">
              <FileCode className="w-8 h-8 text-primary mb-2" />
              <h3 className="font-semibold text-sm">{name}</h3>
              <p className="text-xs text-muted-foreground mt-1">{extension}</p>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );

  return (
    <div className={`min-h-screen flex ${settings.theme === 'dark' ? 'dark' : ''}`}>
      {/* Mobile Sidebar Overlay */}
      {isMobileSidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={() => setIsMobileSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside className={`
        fixed lg:relative z-50 w-72 h-full bg-card border-r border-border 
        flex flex-col transition-transform duration-300 ease-in-out
        ${isMobileSidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
      `}>
        {/* Sidebar Header */}
        <div className="p-4 border-b border-border">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Brain className="w-8 h-8 text-primary" />
              <span className="text-xl font-bold gradient-text">GLM Platform</span>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setIsMobileSidebarOpen(false)}
              className="lg:hidden"
            >
              <X className="w-5 h-5" />
            </Button>
          </div>
          
          <Button 
            onClick={() => { createSession(); setIsMobileSidebarOpen(false); }}
            className="w-full gap-2"
          >
            <Plus className="w-4 h-4" />
            New Chat
          </Button>
        </div>

        {/* Navigation */}
        <div className="p-2 border-b border-border">
          <nav className="space-y-1">
            {[
              { id: 'chat', label: 'Chat', icon: MessageSquare },
              { id: 'training', label: 'Training', icon: Cpu },
              { id: 'export', label: 'Export', icon: Download },
            ].map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => { setActiveView(id as any); setIsMobileSidebarOpen(false); }}
                className={`
                  w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium
                  transition-colors duration-200
                  ${settings.activeView === id
                    ? 'bg-primary/10 text-primary'
                    : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                  }
                `}
              >
                <Icon className="w-4 h-4" />
                {label}
              </button>
            ))}
          </nav>
        </div>

        {/* Sessions List */}
        <ScrollArea className="flex-1 p-2">
          <div className="space-y-1">
            {sessions.map((session) => (
              <div
                key={session.id}
                onClick={() => { setActiveSession(session.id); setActiveView('chat'); setIsMobileSidebarOpen(false); }}
                className={`
                  group flex items-center gap-2 p-3 rounded-lg cursor-pointer
                  transition-colors duration-200
                  ${session.id === activeSessionId && settings.activeView === 'chat'
                    ? 'bg-primary/10 text-primary border border-primary/20' 
                    : 'hover:bg-muted text-muted-foreground hover:text-foreground'
                  }
                `}
              >
                <MessageSquare className="w-4 h-4 flex-shrink-0" />
                <span className="flex-1 truncate text-sm">{session.title}</span>
                <Button
                  variant="ghost"
                  size="icon"
                  className="opacity-0 group-hover:opacity-100 h-6 w-6"
                  onClick={(e) => { e.stopPropagation(); deleteSession(session.id); }}
                >
                  <Trash2 className="w-3 h-3" />
                </Button>
              </div>
            ))}
            
            {sessions.length === 0 && (
              <p className="text-sm text-muted-foreground text-center py-8">
                No conversations yet
              </p>
            )}
          </div>
        </ScrollArea>

        {/* Sidebar Footer */}
        <div className="p-4 border-t border-border space-y-2">
          <Button
            variant="outline"
            className="w-full justify-start gap-2"
            onClick={toggleTheme}
          >
            {settings.theme === 'dark' ? (
              <Sun className="w-4 h-4" />
            ) : (
              <Moon className="w-4 h-4" />
            )}
            {settings.theme === 'dark' ? 'Light Mode' : 'Dark Mode'}
          </Button>
          
          <Badge variant="secondary" className="w-full justify-center py-1">
            GLM Platform v1.0 ✅
          </Badge>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col min-h-screen bg-background">
        {/* Header */}
        <header className="h-16 border-b border-border flex items-center justify-between px-4 bg-card/50 backdrop-blur-sm sticky top-0 z-30">
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setIsMobileSidebarOpen(true)}
              className="lg:hidden"
            >
              <Menu className="w-5 h-5" />
            </Button>
            
            <div>
              <h1 className="font-semibold capitalize">
                {settings.activeView === 'chat' && 'GLM AI Assistant'}
                {settings.activeView === 'training' && 'Model Training'}
                {settings.activeView === 'export' && 'Model Export'}
              </h1>
              <p className="text-xs text-muted-foreground">
                Powered by Z.ai SDK • Free & Open Source
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <Badge variant="outline" className="hidden sm:flex">
              <Sparkles className="w-3 h-3 mr-1" />
              GLM-4 Plus
            </Badge>
            
            <Button variant="ghost" size="icon" onClick={toggleTheme}>
              {settings.theme === 'dark' ? (
                <Sun className="w-5 h-5" />
              ) : (
                <Moon className="w-5 h-5" />
              )}
            </Button>
          </div>
        </header>

        {/* Content Area */}
        <div className="flex-1 overflow-hidden">
          {settings.activeView === 'chat' && (
            <>
              {/* Chat Messages Area */}
              <ScrollArea className="h-full p-4">
                <div className="max-w-3xl mx-auto space-y-6">
                  {activeSession && activeSession.messages.length === 0 && (
                    /* Welcome Screen */
                    <div className="flex flex-col items-center justify-center py-20 text-center">
                      <div className="w-20 h-20 rounded-full bg-primary/10 flex items-center justify-center mb-6 glow-purple">
                        <Sparkles className="w-10 h-10 text-primary" />
                      </div>
                      
                      <h2 className="text-2xl font-bold mb-2 gradient-text">
                        Welcome to GLM Platform
                      </h2>
                      
                      <p className="text-muted-foreground max-w-md mb-8">
                        Your free multimodal AI platform with chat, training, and model export capabilities.
                        No API keys required!
                      </p>

                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-lg">
                        {[
                          { icon: MessageSquare, label: 'Start a conversation', prompt: 'Hello! How can you help me today?' },
                          { icon: Cpu, label: 'Train a model', prompt: 'I want to train a custom language model' },
                          { icon: Image, label: 'Generate an image', prompt: 'Generate an image of a sunset over mountains' },
                          { icon: Brain, label: 'Learn about features', prompt: 'What can you do? Tell me about your capabilities' },
                        ].map(({ icon: Icon, label, prompt }) => (
                          <button
                            key={label}
                            onClick={() => setInputMessage(prompt)}
                            className="flex items-center gap-3 p-3 rounded-lg border border-border hover:border-primary/50 hover:bg-primary/5 transition-all duration-200 text-left"
                          >
                            <Icon className="w-5 h-5 text-primary" />
                            <span className="text-sm">{label}</span>
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Messages */}
                  {activeSession?.messages.map((message) => (
                    <div
                      key={message.id}
                      className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                      <div
                        className={`
                          max-w-[85%] rounded-2xl px-4 py-3
                          ${message.role === 'user'
                            ? 'bg-primary text-primary-foreground rounded-br-md'
                            : 'bg-card border border-border rounded-bl-md'
                          }
                        `}
                      >
                        {message.role === 'assistant' && message.metadata && (
                          <div className="flex items-center gap-2 mb-2 text-xs text-muted-foreground">
                            <Badge variant="secondary" className="text-xs">
                              {message.metadata.model || 'GLM-4'}
                            </Badge>
                            {message.metadata.tokens && (
                              <span>{message.metadata.tokens} tokens</span>
                            )}
                          </div>
                        )}
                        
                        <div className="whitespace-pre-wrap text-sm leading-relaxed">
                          {message.content}
                        </div>
                        
                        <div className={`
                          text-xs mt-2 opacity-60
                          ${message.role === 'user' ? 'text-right' : ''}
                        `}>
                          {new Date(message.timestamp).toLocaleTimeString([], {
                            hour: '2-digit',
                            minute: '2-digit',
                          })}
                        </div>
                      </div>
                    </div>
                  ))}

                  {/* Loading indicator */}
                  {isLoading && (
                    <div className="flex justify-start">
                      <div className="bg-card border border-border rounded-2xl rounded-bl-md px-4 py-3">
                        <div className="flex items-center gap-2">
                          <Loader2 className="w-4 h-4 animate-spin text-primary" />
                          <span className="text-sm text-muted-foreground">GLM is thinking...</span>
                        </div>
                      </div>
                    </div>
                  )}

                  <div ref={messagesEndRef} />
                </div>
              </ScrollArea>

              {/* Input Area */}
              <div className="border-t border-border p-4 bg-card/50 backdrop-blur-sm">
                <div className="max-w-3xl mx-auto">
                  <div className="flex gap-2 items-end">
                    <div className="flex-1 relative">
                      <Input
                        ref={inputRef}
                        value={inputMessage}
                        onChange={(e) => setInputMessage(e.target.value)}
                        onKeyPress={handleKeyPress}
                        placeholder="Type your message..."
                        disabled={isLoading}
                        className="pr-12 min-h-[48px] resize-none rounded-xl"
                        rows={1}
                      />
                    </div>
                    
                    <Button
                      onClick={handleSendMessage}
                      disabled={!inputMessage.trim() || isLoading}
                      size="lg"
                      className="px-6 rounded-xl"
                    >
                      {isLoading ? (
                        <Loader2 className="w-5 h-5 animate-spin" />
                      ) : (
                        <Send className="w-5 h-5" />
                      )}
                    </Button>
                  </div>
                  
                  <p className="text-xs text-muted-foreground text-center mt-2">
                    Free AI powered by Z.ai SDK • No data stored • 100% Private
                  </p>
                </div>
              </div>
            </>
          )}

          {settings.activeView === 'training' && (
            <ScrollArea className="h-full p-4">
              {renderTrainingView()}
            </ScrollArea>
          )}

          {settings.activeView === 'export' && (
            <ScrollArea className="h-full p-4">
              {renderExportView()}
            </ScrollArea>
          )}
        </div>
      </main>
    </div>
  );
}

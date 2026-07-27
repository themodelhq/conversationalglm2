'use client'

import React, { useState, useEffect, useRef } from 'react';
import { 
  MessageSquare, 
  Brain, 
  Download, 
  Settings, 
  BarChart3,
  Zap,
  Eye,
  Mic,
  Video,
  Heart,
  Database,
  Cpu,
  Globe,
  Sparkles,
  Send,
  Plus,
  Trash2,
  Play,
  Pause,
  RotateCcw,
  CheckCircle2,
  XCircle,
  Clock,
  HardDrive,
  MemoryStick,
  Thermometer,
  Activity,
  Layers,
  FileText,
  ImageIcon,
  Music,
  Film,
  ChevronRight,
  ChevronDown,
  Menu,
  X,
  Sun,
  Moon,
  Monitor,
  Copy,
  RefreshCw,
  Upload,
  FolderOpen,
  Terminal,
  Package,
  Rocket,
  Target,
  TrendingUp,
  Users,
  MessageCircle,
  Gauge
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { useAppStore } from '@/store';
import type { Message, TrainingConfig, ModelExport, ExportFormat, QuantizationType } from '@/types';

// Main App Component
export default function Home() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  
  return (
    <div className="min-h-screen bg-background neural-bg">
      {/* Header */}
      <header className="sticky top-0 z-50 w-full glass border-b border-border/50">
        <div className="container mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="icon"
              className="md:hidden"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            >
              {mobileMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </Button>
            <div className="flex items-center gap-2">
              <div className="relative w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-cyan-500 flex items-center justify-center glow-purple">
                <Brain className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold gradient-text">GLM Platform</h1>
                <p className="text-xs text-muted-foreground hidden sm:block">Multimodal AI Suite</p>
              </div>
            </div>
          </div>
          
          <nav className="hidden md:flex items-center gap-1">
            <NavButton icon={<BarChart3 className="h-4 w-4" />} label="Dashboard" active={activeTab === 'dashboard'} onClick={() => setActiveTab('dashboard')} />
            <NavButton icon={<MessageSquare className="h-4 w-4" />} label="Chat" active={activeTab === 'chat'} onClick={() => setActiveTab('chat')} />
            <NavButton icon={<Zap className="h-4 w-4" />} label="Training" active={activeTab === 'training'} onClick={() => setActiveTab('training')} />
            <NavButton icon={<Download className="h-4 w-4" />} label="Export" active={activeTab === 'export'} onClick={() => setActiveTab('export')} />
            <NavButton icon={<Settings className="h-4 w-4" />} label="Settings" active={activeTab === 'settings'} onClick={() => setActiveTab('settings')} />
          </nav>
          
          <div className="flex items-center gap-2">
            <Badge variant="secondary" className="hidden sm:flex gap-1">
              <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
              System Ready
            </Badge>
            <ThemeToggle />
          </div>
        </div>
      </header>

      {/* Mobile Navigation */}
      {mobileMenuOpen && (
        <div className="md:hidden fixed inset-0 z-40 pt-16 glass">
          <nav className="p-4 space-y-2">
            <MobileNavButton icon={<BarChart3 className="h-5 w-5" />} label="Dashboard" onClick={() => { setActiveTab('dashboard'); setMobileMenuOpen(false); }} />
            <MobileNavButton icon={<MessageSquare className="h-5 w-5" />} label="Chat" onClick={() => { setActiveTab('chat'); setMobileMenuOpen(false); }} />
            <MobileNavButton icon={<Zap className="h-5 w-5" />} label="Training" onClick={() => { setActiveTab('training'); setMobileMenuOpen(false); }} />
            <MobileNavButton icon={<Download className="h-5 w-5" />} label="Export" onClick={() => { setActiveTab('export'); setMobileMenuOpen(false); }} />
            <MobileNavButton icon={<Settings className="h-5 w-5" />} label="Settings" onClick={() => { setActiveTab('settings'); setMobileMenuOpen(false); }} />
          </nav>
        </div>
      )}

      {/* Main Content */}
      <main className="container mx-auto px-4 py-6 pb-20">
        {activeTab === 'dashboard' && <DashboardView />}
        {activeTab === 'chat' && <ChatView />}
        {activeTab === 'training' && <TrainingView />}
        {activeTab === 'export' && <ExportView />}
        {activeTab === 'settings' && <SettingsView />}
      </main>

      {/* Footer */}
      <footer className="fixed bottom-0 left-0 right-0 glass border-t border-border/50 py-3 z-40">
        <div className="container mx-auto px-4 flex items-center justify-between text-sm text-muted-foreground">
          <div className="flex items-center gap-4">
            <span>GLM Platform v2.0</span>
            <Separator orientation="vertical" className="h-4" />
            <span className="hidden sm:inline">Multimodal AI Training & Deployment</span>
          </div>
          <div className="flex items-center gap-4">
            <span className="hidden sm:flex items-center gap-1">
              <Cpu className="h-3 w-3" />
              GPU: Available
            </span>
            <span className="flex items-center gap-1">
              <MemoryStick className="h-3 w-3" />
              RAM: 64GB
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
}

// Navigation Components
function NavButton({ icon, label, active, onClick }: { icon: React.ReactNode; label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all ${
        active 
          ? 'bg-gradient-to-r from-purple-500/20 to-cyan-500/20 text-primary border border-purple-500/30' 
          : 'text-muted-foreground hover:text-foreground hover:bg-muted'
      }`}
    >
      {icon}
      {label}
    </button>
  );
}

function MobileNavButton({ icon, label, onClick }: { icon: React.ReactNode; label: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-3 w-full p-3 rounded-lg text-left hover:bg-muted transition-colors"
    >
      {icon}
      <span className="font-medium">{label}</span>
    </button>
  );
}

function ThemeToggle() {
  const { theme, setTheme } = useAppStore();
  
  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
    >
      {theme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
    </Button>
  );
}

// Dashboard View
function DashboardView() {
  const stats = [
    { label: 'Total Conversations', value: '12,847', icon: <MessageCircle className="h-5 w-5" />, change: '+12%', color: 'from-purple-500 to-purple-600' },
    { label: 'Active Trainings', value: '3', icon: <Zap className="h-5 w-5" />, change: '+2', color: 'from-cyan-500 to-cyan-600' },
    { label: 'Models Exported', value: '28', icon: <Package className="h-5 w-5" />, change: '+5', color: 'from-pink-500 to-pink-600' },
    { label: 'GPU Hours Used', value: '1,247', icon: <Activity className="h-5 w-5" />, change: '+8%', color: 'from-orange-500 to-orange-600' },
  ];

  const capabilities = [
    { title: 'Language Model', desc: 'Human-like conversation with context awareness', icon: <MessageSquare />, status: 'ready' },
    { title: 'Vision Encoder', desc: 'Image understanding and analysis', icon: <Eye />, status: 'ready' },
    { title: 'Speech Recognition', desc: 'Transcribe audio to text accurately', icon: <Mic />, status: 'ready' },
    { title: 'Speech Synthesis', desc: 'Natural voice generation with emotion', icon: <Music />, status: 'ready' },
    { title: 'Video Generation', desc: 'Hyper-realistic video creation', icon: <Video />, status: 'beta' },
    { title: 'Emotion AI', desc: 'Recognize and generate emotions', icon: <Heart />, status: 'ready' },
    { title: 'Lip Sync', desc: 'Perfect lip synchronization', icon: <Film />, status: 'beta' },
    { title: 'RAG System', desc: 'Retrieval-augmented generation', icon: <Database />, status: 'ready' },
    { title: 'Long-term Memory', desc: 'Persistent conversation memory', icon: <Brain />, status: 'ready' },
    { title: 'Function Calling', desc: 'Tool use and API integration', icon: <Terminal />, status: 'ready' },
    { title: 'Multi-language', desc: 'Support for 50+ languages', icon: <Globe />, status: 'ready' },
    { title: 'Gesture Generation', desc: 'Natural body animations', icon: <Users />, status: 'beta' },
  ];

  return (
    <div className="space-y-6">
      {/* Hero Section */}
      <div className="relative overflow-hidden rounded-2xl gradient-border">
        <div className="absolute inset-0 bg-gradient-to-br from-purple-500/10 via-transparent to-cyan-500/10" />
        <img 
          src="/images/hero-bg.png" 
          alt="Hero Background" 
          className="absolute inset-0 w-full h-full object-cover opacity-30 dark:opacity-20"
        />
        <div className="relative p-6 md:p-10">
          <div className="max-w-2xl">
            <Badge className="mb-4 bg-gradient-to-r from-purple-500 to-cyan-500 text-white border-0">
              <Sparkles className="h-3 w-3 mr-1" />
              Next Generation AI Platform
            </Badge>
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              <span className="gradient-text">Train. Deploy. Scale.</span>
            </h2>
            <p className="text-muted-foreground text-lg mb-6">
              Complete multimodal AI platform for training GLM models with vision, speech, emotion, 
              video generation, and deployment capabilities.
            </p>
            <div className="flex flex-wrap gap-3">
              <Button className="bg-gradient-to-r from-purple-500 to-cyan-500 text-white hover:opacity-90">
                <Rocket className="h-4 w-4 mr-2" />
                Start Training
              </Button>
              <Button variant="outline" className="border-purple-500/30 hover:bg-purple-500/10">
                <FolderOpen className="h-4 w-4 mr-2" />
                Browse Models
              </Button>
              <Button variant="ghost">
                <FileText className="h-4 w-4 mr-2" />
                Documentation
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((stat) => (
          <Card key={stat.label} className="glass hover:glow-purple transition-all duration-300">
            <CardContent className="p-4 md:p-6">
              <div className="flex items-start justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">{stat.label}</p>
                  <p className="text-2xl md:text-3xl font-bold mt-1">{stat.value}</p>
                </div>
                <div className={`p-2 rounded-lg bg-gradient-to-br ${stat.color} text-white`}>
                  {stat.icon}
                </div>
              </div>
              <div className="flex items-center mt-3 text-sm text-green-500">
                <TrendingUp className="h-3 w-3 mr-1" />
                {stat.change} from last week
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Capabilities Grid */}
      <div>
        <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <Layers className="h-5 w-5 text-purple-500" />
          Platform Capabilities
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {capabilities.map((cap) => (
            <Card key={cap.title} className="glass group hover:border-purple-500/30 transition-all duration-300 cursor-pointer">
              <CardContent className="p-4">
                <div className="flex items-start gap-3">
                  <div className="p-2 rounded-lg bg-gradient-to-br from-purple-500/20 to-cyan-500/20 text-purple-500 group-hover:from-purple-500 group-hover:to-cyan-500 group-hover:text-white transition-all">
                    {React.cloneElement(cap.icon as React.ReactElement, { className: "h-5 w-5" })}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <h4 className="font-semibold truncate">{cap.title}</h4>
                      <Badge variant={cap.status === 'ready' ? 'default' : 'secondary'} className="text-xs">
                        {cap.status}
                      </Badge>
                    </div>
                    <p className="text-sm text-muted-foreground mt-1">{cap.desc}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>

      {/* Quick Actions & Recent Activity */}
      <div className="grid lg:grid-cols-2 gap-6">
        <Card className="glass">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Zap className="h-5 w-5 text-yellow-500" />
              Quick Actions
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <QuickAction 
              icon={<Play className="h-4 w-4" />} 
              label="Start New Training Run" 
              desc="Configure and launch model training"
              color="purple"
            />
            <QuickAction 
              icon={<Upload className="h-4 w-4" />} 
              label="Upload Dataset" 
              desc="Add new training data"
              color="cyan"
            />
            <QuickAction 
              icon={<Download className="h-4 w-4" />} 
              label="Export Model" 
              desc="Download trained model in various formats"
              color="pink"
            />
            <QuickAction 
              icon={<RefreshCw className="h-4 w-4" />} 
              label="Fine-tune Existing Model" 
              desc="Continue training from checkpoint"
              color="orange"
            />
          </CardContent>
        </Card>

        <Card className="glass">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-green-500" />
              System Status
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <SystemStatusItem label="GPU Cluster" value="Online" status="online" detail="8x A100 GPUs available" />
            <SystemStatusItem label="Storage" value="2.4 TB / 10 TB" status="warning" detail="24% used" />
            <SystemStatusItem label="Training Queue" value="3 jobs" status="active" detail="Next: vision_encoder_v2" />
            <SystemStatusItem label="API Server" value="Operational" status="online" detail="Latency: 45ms avg" />
            <Separator />
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">Active Connections</span>
              <span className="font-mono">1,247</span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">Requests Today</span>
              <span className="font-mono">48,293</span>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function QuickAction({ icon, label, desc, color }: { icon: React.ReactNode; label: string; desc: string; color: string }) {
  const colors: Record<string, string> = {
    purple: 'hover:bg-purple-500/10 text-purple-500',
    cyan: 'hover:bg-cyan-500/10 text-cyan-500',
    pink: 'hover:bg-pink-500/10 text-pink-500',
    orange: 'hover:bg-orange-500/10 text-orange-500',
  };
  
  return (
    <button className={`w-full flex items-center gap-3 p-3 rounded-lg transition-colors ${colors[color]} text-left`}>
      <div className="p-2 rounded-md bg-background">{icon}</div>
      <div>
        <p className="font-medium text-sm">{label}</p>
        <p className="text-xs text-muted-foreground">{desc}</p>
      </div>
      <ChevronRight className="h-4 w-4 ml-auto opacity-50" />
    </button>
  );
}

function SystemStatusItem({ label, value, status, detail }: { label: string; value: string; status: string; detail: string }) {
  const statusColors: Record<string, string> = {
    online: 'bg-green-500',
    warning: 'bg-yellow-500',
    active: 'bg-blue-500 animate-pulse',
    error: 'bg-red-500',
  };
  
  return (
    <div className="flex items-center gap-3">
      <span className={`w-2 h-2 rounded-full ${statusColors[status]}`} />
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium">{label}</span>
          <span className="text-sm font-mono">{value}</span>
        </div>
        <p className="text-xs text-muted-foreground">{detail}</p>
      </div>
    </div>
  );
}

// Chat View
function ChatView() {
  const { sessions, activeSessionId, createSession, addMessage, isGenerating, setGenerating } = useAppStore();
  const [inputValue, setInputValue] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  const activeSession = sessions.find(s => s.id === activeSessionId);

  useEffect(() => {
    if (!sessions.length) {
      createSession();
    }
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [activeSession?.messages]);

  const handleSend = async () => {
    if (!inputValue.trim() || !activeSessionId || isGenerating) return;
    
    const userMessage: Message = {
      id: `msg_${Date.now()}`,
      role: 'user',
      content: inputValue,
      timestamp: new Date(),
    };
    
    addMessage(activeSessionId, userMessage);
    setInputValue('');
    setGenerating(true);
    
    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: inputValue,
          sessionId: activeSessionId,
          settings: activeSession?.settings,
        }),
      });
      
      const data = await response.json();
      
      const assistantMessage: Message = {
        id: `msg_${Date.now()}_resp`,
        role: 'assistant',
        content: data.content || data.message || 'Response received',
        timestamp: new Date(),
        metadata: data.metadata,
      };
      
      addMessage(activeSessionId, assistantMessage);
    } catch (error) {
      const errorMessage: Message = {
        id: `msg_${Date.now()}_err`,
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date(),
      };
      addMessage(activeSessionId, errorMessage);
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div className="h-[calc(100vh-10rem)] flex gap-4">
      {/* Session Sidebar */}
      <Card className="w-64 shrink-0 hidden md:flex flex-col glass">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm">Conversations</CardTitle>
            <Button size="sm" variant="ghost" onClick={() => createSession()}>
              <Plus className="h-4 w-4" />
            </Button>
          </div>
        </CardHeader>
        <CardContent className="flex-1 overflow-hidden">
          <ScrollArea className="h-full">
            <div className="space-y-2">
              {sessions.map(session => (
                <button
                  key={session.id}
                  onClick={() => useAppStore.getState().setActiveSession(session.id)}
                  className={`w-full text-left p-2 rounded-lg text-sm truncate transition-colors ${
                    session.id === activeSessionId 
                      ? 'bg-purple-500/20 text-purple-400' 
                      : 'hover:bg-muted text-muted-foreground'
                  }`}
                >
                  {session.title}
                </button>
              ))}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>

      {/* Chat Area */}
      <Card className="flex-1 flex flex-col glass">
        <CardHeader className="pb-3 border-b border-border/50">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-lg">GLM Assistant</CardTitle>
              <CardDescription>Multimodal conversational AI</CardDescription>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="gap-1">
                <Eye className="h-3 w-3" /> Vision
              </Badge>
              <Badge variant="outline" className="gap-1">
                <Mic className="h-3 w-3" /> Voice
              </Badge>
              <Badge variant="outline" className="gap-1">
                <Heart className="h-3 w-3" /> Emotion
              </Badge>
            </div>
          </div>
        </CardHeader>
        
        <CardContent className="flex-1 overflow-hidden p-0">
          <ScrollArea className="h-full p-4 space-y-4">
            {!activeSession?.messages.length && (
              <div className="flex flex-col items-center justify-center h-full text-center text-muted-foreground">
                <div className="w-16 h-16 rounded-full bg-gradient-to-br from-purple-500/20 to-cyan-500/20 flex items-center justify-center mb-4">
                  <MessageSquare className="h-8 w-8 text-purple-500" />
                </div>
                <h3 className="font-semibold text-lg text-foreground">Start a Conversation</h3>
                <p className="text-sm mt-1 max-w-md">
                  Ask me anything! I can help with conversations, analyze images, transcribe audio, 
                  generate content, and much more.
                </p>
                <div className="flex flex-wrap gap-2 mt-4 justify-center">
                  {['Explain quantum computing', 'Generate a story', 'Analyze this image', 'Help me code'].map((prompt) => (
                    <Button
                      key={prompt}
                      variant="outline"
                      size="sm"
                      className="text-xs"
                      onClick={() => setInputValue(prompt)}
                    >
                      {prompt}
                    </Button>
                  ))}
                </div>
              </div>
            )}
            
            {activeSession?.messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                    message.role === 'user'
                      ? 'bg-gradient-to-r from-purple-500 to-purple-600 text-white'
                      : 'glass'
                  }`}
                >
                  <p className="whitespace-pre-wrap text-sm">{message.content}</p>
                  {message.metadata?.emotion && (
                    <div className="flex items-center gap-2 mt-2 text-xs opacity-70">
                      <Heart className="h-3 w-3" />
                      <span>{message.metadata.emotion.primary}</span>
                    </div>
                  )}
                </div>
              </div>
            ))}
            
            {isGenerating && (
              <div className="flex justify-start">
                <div className="glass rounded-2xl px-4 py-3">
                  <div className="flex items-center gap-2">
                    <div className="flex gap-1">
                      <span className="w-2 h-2 rounded-full bg-purple-500 animate-bounce" style={{ animationDelay: '0ms' }} />
                      <span className="w-2 h-2 rounded-full bg-cyan-500 animate-bounce" style={{ animationDelay: '150ms' }} />
                      <span className="w-2 h-2 rounded-full bg-pink-500 animate-bounce" style={{ animationDelay: '300ms' }} />
                    </div>
                    <span className="text-sm text-muted-foreground">Thinking...</span>
                  </div>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </ScrollArea>
        </CardContent>
        
        <div className="p-4 border-t border-border/50">
          <div className="flex gap-2">
            <Input
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
              placeholder="Type your message..."
              className="flex-1"
              disabled={isGenerating}
            />
            <Button 
              onClick={handleSend} 
              disabled={!inputValue.trim() || isGenerating}
              className="bg-gradient-to-r from-purple-500 to-cyan-500 hover:opacity-90"
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>
          <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
            <button className="flex items-center gap-1 hover:text-foreground">
              <ImageIcon className="h-3 w-3" /> Image
            </button>
            <button className="flex items-center gap-1 hover:text-foreground">
              <Mic className="h-3 w-3" /> Voice
            </button>
            <button className="flex items-center gap-1 hover:text-foreground">
              <FileText className="h-3 w-3" /> File
            </button>
            <span className="ml-auto">Press Enter to send</span>
          </div>
        </div>
      </Card>
    </div>
  );
}

// Training View
function TrainingView() {
  const { trainings, addTraining, updateTraining, removeTraining, activeTrainingId, setActiveTraining, trainingMetrics } = useAppStore();
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [newTraining, setNewTraining] = useState<Partial<TrainingConfig>>({
    name: '',
    modelType: 'language',
    hyperparameters: {
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
    hardware: {
      gpus: 1,
      gpuType: 'A100',
      multiNode: false,
      nodes: 1,
      distributedBackend: 'ddp',
    },
  });

  const handleCreateTraining = () => {
    if (!newTraining.name || !newTraining.modelType) return;
    
    const training: TrainingConfig = {
      id: `train_${Date.now()}`,
      name: newTraining.name,
      modelType: newTraining.modelType,
      dataset: {
        name: 'default-dataset',
        path: '/data/datasets/default',
        format: 'jsonl',
        size: 100000,
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
      hyperparameters: newTraining.hyperparameters!,
      hardware: newTraining.hardware!,
      export: {
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
    
    addTraining(training);
    setShowCreateDialog(false);
    setNewTraining({
      name: '',
      modelType: 'language',
      hyperparameters: newTraining.hyperparameters,
      hardware: newTraining.hardware,
    });
  };

  const handleStartTraining = async (id: string) => {
    setActiveTraining(id);
    updateTraining(id, { status: 'preparing' });
    
    // Simulate training progression
    setTimeout(() => updateTraining(id, { status: 'training' }), 2000);
  };

  const handlePauseTraining = (id: string) => {
    updateTraining(id, { status: 'paused' });
  };

  const handleResumeTraining = (id: string) => {
    updateTraining(id, { status: 'training' });
  };

  const activeMetrics = trainingMetrics.slice(-20);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Zap className="h-6 w-6 text-purple-500" />
            Training Pipeline
          </h2>
          <p className="text-muted-foreground">Configure and run model training jobs</p>
        </div>
        <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
          <DialogTrigger asChild>
            <Button className="bg-gradient-to-r from-purple-500 to-cyan-500 text-white">
              <Plus className="h-4 w-4 mr-2" />
              New Training Job
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>Create Training Job</DialogTitle>
              <DialogDescription>Configure your model training parameters</DialogDescription>
            </DialogHeader>
            <div className="space-y-6 py-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Job Name</label>
                <Input
                  value={newTraining.name}
                  onChange={(e) => setNewTraining({ ...newTraining, name: e.target.value })}
                  placeholder="my-glm-model-v1"
                />
              </div>
              
              <div className="space-y-2">
                <label className="text-sm font-medium">Model Type</label>
                <Select
                  value={newTraining.modelType}
                  onValueChange={(v) => setNewTraining({ ...newTraining, modelType: v as any })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="language">Language Model (LLM)</SelectItem>
                    <SelectItem value="speech-recognition">Speech Recognition (ASR)</SelectItem>
                    <SelectItem value="speech-synthesis">Speech Synthesis (TTS)</SelectItem>
                    <SelectItem value="emotion-recognition">Emotion Recognition</SelectItem>
                    <SelectItem value="vision-encoder">Vision Encoder</SelectItem>
                    <SelectItem value="video-generator">Video Generator</SelectItem>
                    <SelectItem value="motion-generator">Motion Generator</SelectItem>
                    <SelectItem value="lip-sync">Lip Sync Model</SelectItem>
                    <SelectItem value="gesture-model">Gesture Model</SelectItem>
                    <SelectItem value="memory-module">Memory Module</SelectItem>
                    <SelectItem value="reward-model">Reward Model</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Learning Rate</label>
                  <Input
                    type="number"
                    value={newTraining.hyperparameters?.learningRate}
                    onChange={(e) => setNewTraining({
                      ...newTraining,
                      hyperparameters: { ...newTraining.hyperparameters!, learningRate: parseFloat(e.target.value) }
                    })}
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Batch Size</label>
                  <Input
                    type="number"
                    value={newTraining.hyperparameters?.batchSize}
                    onChange={(e) => setNewTraining({
                      ...newTraining,
                      hyperparameters: { ...newTraining.hyperparameters!, batchSize: parseInt(e.target.value) }
                    })}
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Epochs</label>
                  <Input
                    type="number"
                    value={newTraining.hyperparameters?.epochs}
                    onChange={(e) => setNewTraining({
                      ...newTraining,
                      hyperparameters: { ...newTraining.hyperparameters!, epochs: parseInt(e.target.value) }
                    })}
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">GPUs</label>
                  <Input
                    type="number"
                    value={newTraining.hardware?.gpus}
                    onChange={(e) => setNewTraining({
                      ...newTraining,
                      hardware: { ...newTraining.hardware!, gpus: parseInt(e.target.value) }
                    })}
                  />
                </div>
              </div>

              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium">Mixed Precision Training</label>
                  <Switch
                    checked={newTraining.hyperparameters?.mixedPrecision}
                    onCheckedChange={(checked) => setNewTraining({
                      ...newTraining,
                      hyperparameters: { ...newTraining.hyperparameters!, mixedPrecision: checked }
                    })}
                  />
                </div>
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium">Gradient Checkpointing</label>
                  <Switch
                    checked={newTraining.hyperparameters?.gradientCheckpointing}
                    onCheckedChange={(checked) => setNewTraining({
                      ...newTraining,
                      hyperparameters: { ...newTraining.hyperparameters!, gradientCheckpointing: checked }
                    })}
                  />
                </div>
              </div>

              <Button onClick={handleCreateTraining} className="w-full bg-gradient-to-r from-purple-500 to-cyan-500 text-white">
                Create Training Job
              </Button>
            </div>
          </DialogContent>
        </Dialog>
      </div>

      {/* Active Training Monitor */}
      {activeTrainingId && (() => {
        const activeTraining = trainings.find(t => t.id === activeTrainingId);
        if (!activeTraining || activeTraining.status !== 'training') return null;
        
        return (
          <Card className="gradient-border glow-purple">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-green-500 animate-pulse" />
                  Active Training: {activeTraining.name}
                </CardTitle>
                <div className="flex gap-2">
                  <Button size="sm" variant="outline" onClick={() => handlePauseTraining(activeTraining.id)}>
                    <Pause className="h-4 w-4 mr-1" />
                    Pause
                  </Button>
                  <Button size="sm" variant="destructive" onClick={() => { updateTraining(activeTraining.id, { status: 'idle' }); setActiveTraining(null); }}>
                    <XCircle className="h-4 w-4 mr-1" />
                    Stop
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-3 gap-6">
                {/* Metrics */}
                <div className="space-y-4">
                  <h4 className="font-medium text-sm text-muted-foreground">Training Metrics</h4>
                  <MetricItem label="Current Loss" value="0.0342" trend="-12%" />
                  <MetricItem label="Learning Rate" value="1.2e-5" />
                  <MetricItem label="Epoch" value="7 / 10" />
                  <MetricItem label="Step" value="14,234 / 20,000" />
                  <MetricItem label="Throughput" value="2,450 samples/s" />
                  <MetricItem label="ETA" value="2h 34m" />
                </div>
                
                {/* GPU Status */}
                <div className="space-y-4">
                  <h4 className="font-medium text-sm text-muted-foreground">GPU Status</h4>
                  {[0, 1, 2, 3].map((gpu) => (
                    <div key={gpu} className="space-y-1">
                      <div className="flex items-center justify-between text-sm">
                        <span>GPU {gpu}</span>
                        <span>{85 + gpu * 3}%</span>
                      </div>
                      <Progress value={85 + gpu * 3} className="h-2" />
                    </div>
                  ))}
                </div>
                
                {/* Loss Chart Placeholder */}
                <div className="space-y-4">
                  <h4 className="font-medium text-sm text-muted-foreground">Loss Curve</h4>
                  <div className="h-48 bg-muted/30 rounded-lg p-4 relative overflow-hidden">
                    <svg viewBox="0 0 200 100" className="w-full h-full">
                      <defs>
                        <linearGradient id="lossGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                          <stop offset="0%" stopColor="#8b5cf6" stopOpacity="0.3" />
                          <stop offset="100%" stopColor="#8b5cf6" stopOpacity="0" />
                        </linearGradient>
                      </defs>
                      <path
                        d="M0,80 Q20,75 40,60 T80,40 T120,25 T160,18 T200,15 L200,100 L0,100 Z"
                        fill="url(#lossGradient)"
                      />
                      <path
                        d="M0,80 Q20,75 40,60 T80,40 T120,25 T160,18 T200,15"
                        fill="none"
                        stroke="#8b5cf6"
                        strokeWidth="2"
                      />
                    </svg>
                    <div className="absolute bottom-2 left-2 right-2 flex justify-between text-xs text-muted-foreground">
                      <span>Step 0</span>
                      <span>Step 14k</span>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        );
      })()}

      {/* Training Jobs List */}
      <div className="grid gap-4">
        {trainings.length === 0 ? (
          <Card className="glass">
            <CardContent className="p-12 text-center">
              <Zap className="h-12 w-12 mx-auto text-muted-foreground/50 mb-4" />
              <h3 className="text-lg font-semibold mb-2">No Training Jobs Yet</h3>
              <p className="text-muted-foreground mb-4">Create your first training job to get started</p>
              <Button onClick={() => setShowCreateDialog(true)} className="bg-gradient-to-r from-purple-500 to-cyan-500 text-white">
                <Plus className="h-4 w-4 mr-2" />
                Create Training Job
              </Button>
            </CardContent>
          </Card>
        ) : (
          trainings.map((training) => (
            <Card key={training.id} className={`glass ${training.id === activeTrainingId ? 'border-purple-500/50' : ''}`}>
              <CardContent className="p-4 md:p-6">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                  <div className="flex items-start gap-4">
                    <div className={`p-3 rounded-xl ${
                      training.status === 'training' ? 'bg-green-500/20 text-green-500' :
                      training.status === 'completed' ? 'bg-blue-500/20 text-blue-500' :
                      training.status === 'failed' ? 'bg-red-500/20 text-red-500' :
                      training.status === 'paused' ? 'bg-yellow-500/20 text-yellow-500' :
                      'bg-gray-500/20 text-gray-500'
                    }`}>
                      {training.status === 'training' ? <Play className="h-5 w-5" /> :
                       training.status === 'completed' ? <CheckCircle2 className="h-5 w-5" /> :
                       training.status === 'failed' ? <XCircle className="h-5 w-5" /> :
                       training.status === 'paused' ? <Pause className="h-5 w-5" /> :
                       <Clock className="h-5 w-5" />}
                    </div>
                    <div>
                      <h3 className="font-semibold">{training.name}</h3>
                      <div className="flex items-center gap-2 mt-1">
                        <Badge variant="secondary" className="text-xs">{training.modelType}</Badge>
                        <span className="text-xs text-muted-foreground">
                          {training.hardware.gpus}x {training.hardware.gpuType}
                        </span>
                        <span className="text-xs text-muted-foreground">
                          {training.createdAt.toLocaleDateString()}
                        </span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    {training.status === 'idle' && (
                      <Button size="sm" onClick={() => handleStartTraining(training.id)}>
                        <Play className="h-4 w-4 mr-1" />
                        Start
                      </Button>
                    )}
                    {training.status === 'training' && (
                      <Button size="sm" variant="outline" onClick={() => handlePauseTraining(training.id)}>
                        <Pause className="h-4 w-4 mr-1" />
                        Pause
                      </Button>
                    )}
                    {training.status === 'paused' && (
                      <Button size="sm" onClick={() => handleResumeTraining(training.id)}>
                        <RotateCcw className="h-4 w-4 mr-1" />
                        Resume
                      </Button>
                    )}
                    <Button size="sm" variant="ghost" onClick={() => removeTraining(training.id)}>
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
                
                {(training.status === 'training' || training.status === 'paused') && (
                  <div className="mt-4 space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span>Progress</span>
                      <span>71%</span>
                    </div>
                    <Progress value={71} className="h-2" />
                  </div>
                )}
              </CardContent>
            </Card>
          ))
        )}
      </div>
    </div>
  );
}

function MetricItem({ label, value, trend }: { label: string; value: string; trend?: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-sm text-muted-foreground">{label}</span>
      <div className="flex items-center gap-2">
        <span className="font-mono text-sm">{value}</span>
        {trend && (
          <span className={`text-xs ${trend.startsWith('-') ? 'text-green-500' : 'text-red-500'}`}>
            {trend}
          </span>
        )}
      </div>
    </div>
  );
}

// Export View
function ExportView() {
  const { exports: exportedModels, addExport, isExporting, setExporting } = useAppStore();
  const [selectedModel, setSelectedModel] = useState('');
  const [selectedFormats, setSelectedFormats] = useState<ExportFormat[]>(['pytorch']);
  const [quantization, setQuantization] = useState<QuantizationType>('fp16');
  
  const formats: { id: ExportFormat; name: string; desc: string; icon: React.ReactNode }[] = [
    { id: 'pytorch', name: 'PyTorch', desc: 'Native PyTorch format', icon: <Zap className="h-4 w-4" /> },
    { id: 'onnx', name: 'ONNX', desc: 'Universal inference format', icon: <Globe className="h-4 w-4" /> },
    { id: 'tensorrt', name: 'TensorRT', desc: 'NVIDIA optimized', icon: <Cpu className="h-4 w-4" /> },
    { id: 'gguf', name: 'GGUF', desc: 'llama.cpp compatible', icon: <Package className="h-4 w-4" /> },
    { id: 'safetensors', name: 'SafeTensors', desc: 'Safe tensor format', icon: <Shield className="h-4 w-4" /> },
  ];
  
  const quantOptions: { id: QuantizationType; name: string; desc: string }[] = [
    { id: 'fp32', name: 'FP32', desc: 'Full precision (baseline)' },
    { id: 'fp16', name: 'FP16', desc: 'Half precision (recommended)' },
    { id: 'int8', name: 'INT8', desc: '8-bit quantization' },
    { id: 'int4', name: 'INT4', desc: '4-bit quantization (extreme)' },
    { id: 'awq', name: 'AWQ', desc: 'Activation-aware quantization' },
    { id: 'gptq', name: 'GPTQ', desc: 'GPT quantization' },
  ];

  const handleExport = async () => {
    if (!selectedModel || selectedFormats.length === 0) return;
    
    setExporting(true);
    
    for (const format of selectedFormats) {
      const exp: ModelExport = {
        id: `exp_${Date.now()}_${format}`,
        modelName: selectedModel,
        format,
        path: `/exports/${selectedModel}/${format}`,
        size: 0,
        quantization,
        status: 'exporting',
        createdAt: new Date(),
      };
      
      addExport(exp);
      
      // Simulate export process
      setTimeout(() => {
        addExport({
          ...exp,
          status: 'completed',
          size: Math.floor(Math.random() * 10000000000),
          metrics: {
            latency_ms: Math.floor(Math.random() * 100) + 10,
            throughput_samples: Math.floor(Math.random() * 1000) + 100,
            memory_mb: Math.floor(Math.random() * 8000) + 1000,
            compression_ratio: format === 'gguf' ? 4 : format === 'onnx' ? 1.2 : 1,
          },
        });
      }, 2000 + Math.random() * 3000);
    }
    
    setTimeout(() => setExporting(false), 5000);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Download className="h-6 w-6 text-cyan-500" />
            Model Export
          </h2>
          <p className="text-muted-foreground">Export trained models in various formats</p>
        </div>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Export Configuration */}
        <Card className="lg:col-span-2 glass">
          <CardHeader>
            <CardTitle>Export Configuration</CardTitle>
            <CardDescription>Select model and export options</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-2">
              <label className="text-sm font-medium">Model to Export</label>
              <Select value={selectedModel} onValueChange={setSelectedModel}>
                <SelectTrigger>
                  <SelectValue placeholder="Select a trained model" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="glm-4-base">GLM-4 Base (9B)</SelectItem>
                  <SelectItem value="glm-4-chat">GLM-4 Chat (9B)</SelectItem>
                  <SelectItem value="glm-4v">GLM-4V Vision (9B)</SelectItem>
                  <SelectItem value="glm-4-speech">GLM-4 Speech ASR</SelectItem>
                  <SelectItem value="glm-4-tts">GLM-4 TTS Voice</SelectItem>
                  <SelectItem value="vision-encoder-v2">Vision Encoder V2</SelectItem>
                  <SelectItem value="video-gen-base">Video Generator Base</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-3">
              <label className="text-sm font-medium">Export Formats</label>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {formats.map((fmt) => (
                  <button
                    key={fmt.id}
                    onClick={() => {
                      if (selectedFormats.includes(fmt.id)) {
                        setSelectedFormats(selectedFormats.filter(f => f !== fmt.id));
                      } else {
                        setSelectedFormats([...selectedFormats, fmt.id]);
                      }
                    }}
                    className={`flex items-center gap-3 p-3 rounded-lg border transition-all ${
                      selectedFormats.includes(fmt.id)
                        ? 'border-purple-500 bg-purple-500/10 text-purple-500'
                        : 'border-border hover:border-muted-foreground/30'
                    }`}
                  >
                    {fmt.icon}
                    <div className="text-left">
                      <p className="font-medium text-sm">{fmt.name}</p>
                      <p className="text-xs text-muted-foreground">{fmt.desc}</p>
                    </div>
                    {selectedFormats.includes(fmt.id) && (
                      <CheckCircle2 className="h-4 w-4 ml-auto" />
                    )}
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-3">
              <label className="text-sm font-medium">Quantization</label>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                {quantOptions.map((q) => (
                  <button
                    key={q.id}
                    onClick={() => setQuantization(q.id)}
                    className={`p-2 rounded-lg border text-sm transition-all ${
                      quantization === q.id
                        ? 'border-cyan-500 bg-cyan-500/10 text-cyan-500'
                        : 'border-border hover:border-muted-foreground/30'
                    }`}
                  >
                    <p className="font-medium">{q.name}</p>
                    <p className="text-xs text-muted-foreground">{q.desc}</p>
                  </button>
                ))}
              </div>
            </div>

            <Button
              onClick={handleExport}
              disabled={!selectedModel || selectedFormats.length === 0 || isExporting}
              className="w-full bg-gradient-to-r from-cyan-500 to-purple-500 text-white"
            >
              {isExporting ? (
                <>
                  <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  Exporting...
                </>
              ) : (
                <>
                  <Download className="h-4 w-4 mr-2" />
                  Start Export ({selectedFormats.length} format{selectedFormats.length > 1 ? 's' : ''})
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Export History */}
        <Card className="glass">
          <CardHeader>
            <CardTitle>Export History</CardTitle>
            <CardDescription>Recent exports</CardDescription>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[400px]">
              <div className="space-y-3">
                {exportedModels.length === 0 ? (
                  <p className="text-sm text-muted-foreground text-center py-8">
                    No exports yet
                  </p>
                ) : (
                  exportedModels.map((exp) => (
                    <div key={exp.id} className="p-3 rounded-lg bg-muted/30 space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="font-medium text-sm truncate">{exp.modelName}</span>
                        <Badge variant={exp.status === 'completed' ? 'default' : 'secondary'} className="text-xs">
                          {exp.status}
                        </Badge>
                      </div>
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <Badge variant="outline" className="text-xs">{exp.format.toUpperCase()}</Badge>
                        <Badge variant="outline" className="text-xs">{exp.quantization.toUpperCase()}</Badge>
                      </div>
                      {exp.metrics && (
                        <div className="grid grid-cols-2 gap-2 text-xs">
                          <div><span className="text-muted-foreground">Latency:</span> {exp.metrics.latency_ms}ms</div>
                          <div><span className="text-muted-foreground">Size:</span> {(exp.size / 1e9).toFixed(2)}GB</div>
                        </div>
                      )}
                    </div>
                  ))
                )}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      </div>

      {/* Format Comparison Table */}
      <Card className="glass">
        <CardHeader>
          <CardTitle>Format Comparison</CardTitle>
          <CardDescription>Choose the best format for your deployment needs</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left p-3 font-medium">Format</th>
                  <th className="text-left p-3 font-medium">Best For</th>
                  <th className="text-left p-3 font-medium">Size</th>
                  <th className="text-left p-3 font-medium">Speed</th>
                  <th className="text-left p-3 font-medium">Compatibility</th>
                </tr>
              </thead>
              <tbody>
                <TableRow format="PyTorch" bestFor="Training & Research" size="1x" speed="Baseline" compat="PyTorch ecosystem" />
                <TableRow format="ONNX" bestFor="Cross-platform deploy" size="0.9x" speed="1.2x" compat="Universal" highlight />
                <TableRow format="TensorRT" bestFor="NVIDIA GPU inference" size="0.8x" speed="2-3x" compat="NVIDIA only" />
                <TableRow format="GGUF" bestFor="Local/CPU inference" size="0.25-0.5x" speed="0.8x" compat="llama.cpp" />
                <TableRow format="SafeTensors" bestFor="Safe storage/sharing" size="1x" speed="Baseline" compat="HuggingFace" />
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function TableRow({ format, bestFor, size, speed, compat, highlight }: { format: string; bestFor: string; size: string; speed: string; compat: string; highlight?: boolean }) {
  return (
    <tr className={`border-b border-border/50 ${highlight ? 'bg-purple-500/5' : ''}`}>
      <td className="p-3 font-medium">{format}</td>
      <td className="p-3 text-muted-foreground">{bestFor}</td>
      <td className="p-3"><Badge variant="outline">{size}</Badge></td>
      <td className="p-3"><Badge variant="outline">{speed}</Badge></td>
      <td className="p-3 text-muted-foreground">{compat}</td>
    </tr>
  );
}

// Shield icon for SafeTensors
function Shield(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
    </svg>
  );
}

// Settings View
function SettingsView() {
  const { theme, setTheme } = useAppStore();
  
  return (
    <div className="space-y-6 max-w-4xl">
      <div>
        <h2 className="text-2xl font-bold flex items-center gap-2">
          <Settings className="h-6 w-6 text-orange-500" />
          Settings
        </h2>
        <p className="text-muted-foreground">Configure platform settings</p>
      </div>

      <Tabs defaultValue="general">
        <TabsList className="grid grid-cols-2 md:grid-cols-4 w-full">
          <TabsTrigger value="general">General</TabsTrigger>
          <TabsTrigger value="models">Models</TabsTrigger>
          <TabsTrigger value="hardware">Hardware</TabsTrigger>
          <TabsTrigger value="api">API Keys</TabsTrigger>
        </TabsList>
        
        <TabsContent value="general" className="space-y-4 mt-4">
          <Card className="glass">
            <CardHeader>
              <CardTitle>Appearance</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Theme</p>
                  <p className="text-sm text-muted-foreground">Choose your preferred theme</p>
                </div>
                <div className="flex gap-2">
                  <Button
                    variant={theme === 'light' ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setTheme('light')}
                  >
                    <Sun className="h-4 w-4 mr-1" />
                    Light
                  </Button>
                  <Button
                    variant={theme === 'dark' ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setTheme('dark')}
                  >
                    <Moon className="h-4 w-4 mr-1" />
                    Dark
                  </Button>
                  <Button
                    variant={theme === 'system' ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setTheme('system')}
                  >
                    <Monitor className="h-4 w-4 mr-1" />
                    System
                  </Button>
                </div>
              </div>
              
              <Separator />
              
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Animations</p>
                  <p className="text-sm text-muted-foreground">Enable UI animations</p>
                </div>
                <Switch defaultChecked />
              </div>
              
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Sound Effects</p>
                  <p className="text-sm text-muted-foreground">Play sounds for notifications</p>
                </div>
                <Switch />
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="models" className="space-y-4 mt-4">
          <Card className="glass">
            <CardHeader>
              <CardTitle>Default Model Settings</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Default Language Model</label>
                <Select defaultValue="glm-4-chat">
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="glm-4-chat">GLM-4 Chat</SelectItem>
                    <SelectItem value="glm-4-plus">GLM-4 Plus</SelectItem>
                    <SelectItem value="glm-4-long">GLM-4 Long Context</SelectItem>
                    <SelectItem value="glm-4v">GLM-4 Vision</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="space-y-2">
                <label className="text-sm font-medium">Default Temperature: 0.7</label>
                <Slider defaultValue={[0.7]} max={2} step={0.1} />
              </div>
              
              <div className="space-y-2">
                <label className="text-sm font-medium">Max Tokens: 4096</label>
                <Slider defaultValue={[4096]} max={32000} step={256} />
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="hardware" className="space-y-4 mt-4">
          <Card className="glass">
            <CardHeader>
              <CardTitle>Hardware Configuration</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 rounded-lg bg-muted/30">
                  <div className="flex items-center gap-2 mb-2">
                    <Cpu className="h-4 w-4 text-green-500" />
                    <span className="font-medium">GPU 0</span>
                  </div>
                  <p className="text-sm text-muted-foreground">NVIDIA A100 80GB</p>
                  <Progress value={23} className="mt-2 h-2" />
                  <p className="text-xs text-muted-foreground mt-1">23% utilized</p>
                </div>
                
                <div className="p-4 rounded-lg bg-muted/30">
                  <div className="flex items-center gap-2 mb-2">
                    <Cpu className="h-4 w-4 text-green-500" />
                    <span className="font-medium">GPU 1</span>
                  </div>
                  <p className="text-sm text-muted-foreground">NVIDIA A100 80GB</p>
                  <Progress value={67} className="mt-2 h-2" />
                  <p className="text-xs text-muted-foreground mt-1">67% utilized</p>
                </div>
              </div>
              
              <Separator />
              
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span>Total GPU Memory</span>
                  <span className="font-mono">160 GB</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span>Used GPU Memory</span>
                  <span className="font-mono">72 GB (45%)</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span>System RAM</span>
                  <span className="font-mono">512 GB</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span>Storage</span>
                  <span className="font-mono">2.4 TB / 10 TB</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="api" className="space-y-4 mt-4">
          <Card className="glass">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>API Keys</CardTitle>
                <Button size="sm" className="bg-gradient-to-r from-purple-500 to-cyan-500 text-white">
                  <Plus className="h-4 w-4 mr-1" />
                  Generate Key
                </Button>
              </div>
              <CardDescription>Manage your API access keys</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <ApiKeyRow name="Production Key" keyPreview="glm_prod_****...a8f2" lastUsed="2 minutes ago" usageCount={12453} />
                <ApiKeyRow name="Development Key" keyPreview="glm_dev_****...b3c1" lastUsed="1 hour ago" usageCount={892} />
                <ApiKeyRow name="Testing Key" keyPreview="glm_test_****...d7e4" lastUsed="Yesterday" usageCount={45} />
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

function ApiKeyRow({ name, keyPreview, lastUsed, usageCount }: { name: string; keyPreview: string; lastUsed: string; usageCount: number }) {
  return (
    <div className="flex items-center justify-between p-3 rounded-lg bg-muted/30">
      <div>
        <p className="font-medium text-sm">{name}</p>
        <p className="text-xs text-muted-foreground font-mono">{keyPreview}</p>
      </div>
      <div className="flex items-center gap-4">
        <div className="text-right hidden sm:block">
          <p className="text-xs text-muted-foreground">Last used: {lastUsed}</p>
          <p className="text-xs text-muted-foreground">{usageCount.toLocaleString()} calls</p>
        </div>
        <div className="flex gap-1">
          <Button size="sm" variant="ghost">
            <Copy className="h-4 w-4" />
          </Button>
          <Button size="sm" variant="ghost" className="text-destructive">
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}

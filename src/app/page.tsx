'use client'

import React, { useState, useRef, useEffect } from 'react';
import { useAppStore, BACKEND_URL } from '@/store';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { 
  MessageSquare, 
  Send, 
  Plus, 
  Trash2, 
  Settings, 
  Sparkles,
  Image,
  Brain,
  Download,
  Moon,
  Sun,
  Menu,
  X,
  Loader2
} from 'lucide-react';

export default function GLMPlatform() {
  const {
    sessions,
    activeSessionId,
    isLoading,
    settings,
    createSession,
    deleteSession,
    setActiveSession,
    addMessage,
    updateSessionTitle,
    setLoading,
    toggleSidebar,
    setSettings,
  } = useAppStore();

  const [inputMessage, setInputMessage] = useState('');
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

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

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || !activeSessionId) return;

    const userMessage = {
      id: `msg-${Date.now()}`,
      role: 'user' as const,
      content: inputMessage.trim(),
      timestamp: new Date(),
    };

    // Add user message
    addMessage(activeSessionId, userMessage);
    
    // Update session title based on first message
    if (activeSession && activeSession.messages.length === 0) {
      updateSessionTitle(activeSessionId, inputMessage.trim().slice(0, 30) + '...');
    }

    setInputMessage('');
    setLoading(true);

    try {
      // Call backend API
      const response = await fetch(`${BACKEND_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage.content,
          sessionId: activeSessionId,
          settings: activeSession?.settings,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (data.success) {
        const assistantMessage = {
          id: `msg-${Date.now()}-resp`,
          role: 'assistant' as const,
          content: data.data.content,
          timestamp: new Date(),
          metadata: data.data.metadata,
        };

        addMessage(activeSessionId, assistantMessage);
      } else {
        throw new Error(data.error || 'Failed to get response');
      }
    } catch (error: any) {
      console.error('Error sending message:', error);
      
      // Add error message
      const errorMessage = {
        id: `msg-${Date.now()}-error`,
        role: 'assistant' as const,
        content: `Sorry, I encountered an error: ${error.message}. Please try again.`,
        timestamp: new Date(),
      };
      
      addMessage(activeSessionId, errorMessage);
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

        {/* Sessions List */}
        <ScrollArea className="flex-1 p-2">
          <div className="space-y-1">
            {sessions.map((session) => (
              <div
                key={session.id}
                onClick={() => { setActiveSession(session.id); setIsMobileSidebarOpen(false); }}
                className={`
                  group flex items-center gap-2 p-3 rounded-lg cursor-pointer
                  transition-colors duration-200
                  ${session.id === activeSessionId 
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
                No conversations yet. Start a new chat!
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
            Connected to Backend ✅
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
              <h1 className="font-semibold">GLM AI Assistant</h1>
              <p className="text-xs text-muted-foreground">
                Powered by Z.ai • {BACKEND_URL}
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

        {/* Chat Messages Area */}
        <div className="flex-1 overflow-hidden">
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
                    Your free multimodal AI assistant. I can help with chat, image generation, 
                    code, and more. No API keys required!
                  </p>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-lg">
                    {[
                      { icon: MessageSquare, label: 'Start a conversation', prompt: 'Hello! How can you help me today?' },
                      { icon: Image, label: 'Generate an image', prompt: 'Generate an image of a sunset over mountains' },
                      { icon: Brain, label: 'Get coding help', prompt: 'Help me write a Python function to sort a list' },
                      { icon: Download, label: 'Learn about features', prompt: 'What can you do? Tell me about your capabilities' },
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
                    {/* Message metadata for assistant */}
                    {message.role === 'assistant' && message.metadata && (
                      <div className="flex items-center gap-2 mb-2 text-xs text-muted-foreground">
                        <Badge variant="secondary" className="text-xs">
                          {message.metadata.model || 'GLM-4'}
                        </Badge>
                        {message.metadata.tokens && (
                          <span>{message.metadata.tokens} tokens</                        )}
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
        </div>

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
      </main>
    </div>
  );
}

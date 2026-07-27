import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import type { 
  Message, 
  ChatSession, 
  SessionSettings, 
  TrainingConfig,
  TrainingMetrics,
  ModelExport,
  DashboardStats 
} from '@/types';

interface AppState {
  // Chat State
  sessions: ChatSession[];
  activeSessionId: string | null;
  isGenerating: boolean;
  
  // Training State
  trainings: TrainingConfig[];
  activeTrainingId: string | null;
  trainingMetrics: TrainingMetrics[];
  
  // Export State
  exports: ModelExport[];
  isExporting: boolean;
  
  // Dashboard
  dashboardStats: DashboardStats;
  
  // UI State
  sidebarOpen: boolean;
  theme: 'light' | 'dark' | 'system';
  
  // Actions
  createSession: (model?: string) => string;
  deleteSession: (id: string) => void;
  setActiveSession: (id: string) => void;
  addMessage: (sessionId: string, message: Message) => void;
  updateMessage: (sessionId: string, messageId: string, updates: Partial<Message>) => void;
  clearMessages: (sessionId: string) => void;
  setGenerating: (value: boolean) => void;
  updateSessionSettings: (sessionId: string, settings: Partial<SessionSettings>) => void;
  
  // Training Actions
  addTraining: (training: TrainingConfig) => void;
  updateTraining: (id: string, updates: Partial<TrainingConfig>) => void;
  removeTraining: (id: string) => void;
  setActiveTraining: (id: string | null) => void;
  addTrainingMetric: (metric: TrainingMetrics) => void;
  clearTrainingMetrics: () => void;
  
  // Export Actions
  addExport: (exp: ModelExport) => void;
  updateExport: (id: string, updates: Partial<ModelExport>) => void;
  removeExport: (id: string) => void;
  setExporting: (value: boolean) => void;
  
  // Dashboard Actions
  setDashboardStats: (stats: DashboardStats) => void;
  
  // UI Actions
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
  setTheme: (theme: 'light' | 'dark' | 'system') => void;
}

export const useAppStore = create<AppState>()(
  devtools(
    persist(
      (set, get) => ({
        // Initial State
        sessions: [],
        activeSessionId: null,
        isGenerating: false,
        
        trainings: [],
        activeTrainingId: null,
        trainingMetrics: [],
        
        exports: [],
        isExporting: false,
        
        dashboardStats: {
          totalConversations: 0,
          totalMessages: 0,
          activeTrainings: 0,
          completedModels: 0,
          gpuHoursUsed: 0,
          storageUsed: 0,
          apiCallsToday: 0,
          activeUsers: 0,
        },
        
        sidebarOpen: true,
        theme: 'dark',
        
        // Chat Actions
        createSession: (model = 'glm-4') => {
          const id = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
          const newSession: ChatSession = {
            id,
            title: 'New Conversation',
            messages: [],
            createdAt: new Date(),
            updatedAt: new Date(),
            model,
            settings: {
              temperature: 0.7,
              maxTokens: 4096,
              systemPrompt: 'You are GLM, a helpful AI assistant.',
              enableMemory: true,
              enableRAG: true,
              enableVision: true,
              enableVoice: false,
              emotionMode: 'expressive',
            },
          };
          
          set(state => ({
            sessions: [newSession, ...state.sessions],
            activeSessionId: id,
          }));
          
          return id;
        },
        
        deleteSession: (id) => {
          set(state => ({
            sessions: state.sessions.filter(s => s.id !== id),
            activeSessionId: state.activeSessionId === id ? null : state.activeSessionId,
          }));
        },
        
        setActiveSession: (id) => {
          set({ activeSessionId: id });
        },
        
        addMessage: (sessionId, message) => {
          set(state => ({
            sessions: state.sessions.map(s => {
              if (s.id === sessionId) {
                const updatedMessages = [...s.messages, message];
                return {
                  ...s,
                  messages: updatedMessages,
                  title: s.messages.length === 0 ? message.content.slice(0, 50) + '...' : s.title,
                  updatedAt: new Date(),
                };
              }
              return s;
            }),
          }));
        },
        
        updateMessage: (sessionId, messageId, updates) => {
          set(state => ({
            sessions: state.sessions.map(s => {
              if (s.id === sessionId) {
                return {
                  ...s,
                  messages: s.messages.map(m => 
                    m.id === messageId ? { ...m, ...updates } : m
                  ),
                };
              }
              return s;
            }),
          }));
        },
        
        clearMessages: (sessionId) => {
          set(state => ({
            sessions: state.sessions.map(s => {
              if (s.id === sessionId) {
                return { ...s, messages: [], title: 'New Conversation' };
              }
              return s;
            }),
          }));
        },
        
        setGenerating: (value) => {
          set({ isGenerating: value });
        },
        
        updateSessionSettings: (sessionId, settings) => {
          set(state => ({
            sessions: state.sessions.map(s => {
              if (s.id === sessionId) {
                return { ...s, settings: { ...s.settings, ...settings } };
              }
              return s;
            }),
          }));
        },
        
        // Training Actions
        addTraining: (training) => {
          set(state => ({ trainings: [...state.trainings, training] }));
        },
        
        updateTraining: (id, updates) => {
          set(state => ({
            trainings: state.trainings.map(t =>
              t.id === id ? { ...t, ...updates } : t
            ),
          }));
        },
        
        removeTraining: (id) => {
          set(state => ({
            trainings: state.trainings.filter(t => t.id !== id),
            activeTrainingId: state.activeTrainingId === id ? null : state.activeTrainingId,
          }));
        },
        
        setActiveTraining: (id) => {
          set({ activeTrainingId: id });
        },
        
        addTrainingMetric: (metric) => {
          set(state => ({ trainingMetrics: [...state.trainingMetrics, metric] }));
        },
        
        clearTrainingMetrics: () => {
          set({ trainingMetrics: [] });
        },
        
        // Export Actions
        addExport: (exp) => {
          set(state => ({ exports: [...state.exports, exp] }));
        },
        
        updateExport: (id, updates) => {
          set(state => ({
            exports: state.exports.map(e =>
              e.id === id ? { ...e, ...updates } : e
            ),
          }));
        },
        
        removeExport: (id) => {
          set(state => ({
            exports: state.exports.filter(e => e.id !== id),
          }));
        },
        
        setExporting: (value) => {
          set({ isExporting: value });
        },
        
        // Dashboard Actions
        setDashboardStats: (stats) => {
          set({ dashboardStats: stats });
        },
        
        // UI Actions
        toggleSidebar: () => {
          set(state => ({ sidebarOpen: !state.sidebarOpen }));
        },
        
        setSidebarOpen: (open) => {
          set({ sidebarOpen: open });
        },
        
        setTheme: (theme) => {
          set({ theme });
        },
      }),
      {
        name: 'glm-platform-storage',
        partialize: (state) => ({
          sessions: state.sessions,
          activeSessionId: state.activeSessionId,
          theme: state.theme,
          sidebarOpen: state.sidebarOpen,
        }),
      }
    )
  )
);

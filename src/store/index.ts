import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { Message, ChatSession, ChatSettings, TrainingJob, ExportJob, AppSettings, ViewType } from '@/types';

// Backend URL configuration
// - If NEXT_PUBLIC_API_URL is set to an external URL (e.g., Render backend), use it
// - If empty or not set, use local API routes (same-origin, no CORS issues)
export const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || '';

// Helper function to get the correct API endpoint
// This ensures we use local routes by default (avoids CORS) 
// and only use external backend when explicitly configured
export function getApiUrl(path: string): string {
  if (BACKEND_URL && BACKEND_URL.startsWith('http')) {
    // External backend mode - for when frontend and backend are on different domains
    return `${BACKEND_URL}${path}`;
  }
  // Local mode - uses same-origin API routes (no CORS issues)
  return path;
}

// Check if we're using external backend mode
export function isExternalBackendMode(): boolean {
  return !!(BACKEND_URL && BACKEND_URL.startsWith('http'));
}

interface AppState {
  // Settings
  settings: AppSettings;
  setSettings: (settings: Partial<AppSettings>) => void;
  setActiveView: (view: ViewType) => void;
  toggleSidebar: () => void;
  
  // Chat
  sessions: ChatSession[];
  activeSessionId: string | null;
  isLoading: boolean;
  
  createSession: () => string;
  deleteSession: (id: string) => void;
  setActiveSession: (id: string) => void;
  addMessage: (sessionId: string, message: Message) => void;
  updateSessionTitle: (sessionId: string, title: string) => void;
  setLoading: (loading: boolean) => void;
  
  // Training
  trainingJobs: TrainingJob[];
  addTrainingJob: (job: TrainingJob) => void;
  updateTrainingJob: (id: string, updates: Partial<TrainingJob>) => void;
  removeTrainingJob: (id: string) => void;
  
  // Export
  exportJobs: ExportJob[];
  addExportJob: (job: ExportJob) => void;
  updateExportJob: (id: string, updates: Partial<ExportJob>) => void;
  removeExportJob: (id: string) => void;
}

const defaultChatSettings: ChatSettings = {
  model: 'glm-4-plus',
  temperature: 0.7,
  maxTokens: 4096,
  systemPrompt: `You are GLM, a helpful AI assistant with multimodal capabilities. 
You can help with:
- Natural conversation and question answering
- Image analysis and understanding
- Code generation and debugging
- Creative writing and content creation
- Data analysis and problem solving

Be helpful, accurate, and engaging in your responses.`,
};

export const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      // Settings
      settings: {
        theme: 'dark',
        sidebarOpen: true,
        activeView: 'chat',
      },
      
      setSettings: (newSettings) =>
        set((state) => ({
          settings: { ...state.settings, ...newSettings },
        })),
      
      setActiveView: (view) =>
        set((state) => ({
          settings: { ...state.settings, activeView: view },
        })),
      
      toggleSidebar: () =>
        set((state) => ({
          settings: { ...state.settings, sidebarOpen: !state.settings.sidebarOpen },
        })),
      
      // Chat
      sessions: [],
      activeSessionId: null,
      isLoading: false,
      
      createSession: () => {
        const id = `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        const newSession: ChatSession = {
          id,
          title: 'New Conversation',
          messages: [],
          createdAt: new Date(),
          updatedAt: new Date(),
          settings: defaultChatSettings,
        };
        
        set((state) => ({
          sessions: [newSession, ...state.sessions],
          activeSessionId: id,
        }));
        
        return id;
      },
      
      deleteSession: (id) =>
        set((state) => {
          const filtered = state.sessions.filter((s) => s.id !== id);
          return {
            sessions: filtered,
            activeSessionId:
              state.activeSessionId === id
                ? filtered[0]?.id || null
                : state.activeSessionId,
          };
        }),
      
      setActiveSession: (id) => set({ activeSessionId: id }),
      
      addMessage: (sessionId, message) =>
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === sessionId
              ? {
                  ...s,
                  messages: [...s.messages, message],
                  updatedAt: new Date(),
                }
              : s
          ),
        })),
      
      updateSessionTitle: (sessionId, title) =>
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === sessionId ? { ...s, title } : s
          ),
        })),
      
      setLoading: (loading) => set({ isLoading: loading }),
      
      // Training
      trainingJobs: [],
      
      addTrainingJob: (job) =>
        set((state) => ({
          trainingJobs: [...state.trainingJobs, job],
        })),
      
      updateTrainingJob: (id, updates) =>
        set((state) => ({
          trainingJobs: state.trainingJobs.map((j) =>
            j.id === id ? { ...j, ...updates } : j
          ),
        })),
      
      removeTrainingJob: (id) =>
        set((state) => ({
          trainingJobs: state.trainingJobs.filter((j) => j.id !== id),
        })),
      
      // Export
      exportJobs: [],
      
      addExportJob: (job) =>
        set((state) => ({
          exportJobs: [...state.exportJobs, job],
        })),
      
      updateExportJob: (id, updates) =>
        set((state) => ({
          exportJobs: state.exportJobs.map((j) =>
            j.id === id ? { ...j, ...updates } : j
          ),
        })),
      
      removeExportJob: (id) =>
        set((state) => ({
          exportJobs: state.exportJobs.filter((j) => j.id !== id),
        })),
    }),
    {
      name: 'glm-platform-storage',
      partialize: (state) => ({
        settings: state.settings,
        sessions: state.sessions,
        activeSessionId: state.activeSessionId,
      }),
    }
  )
);

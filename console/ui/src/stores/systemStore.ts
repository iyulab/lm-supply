import { create } from 'zustand';
import type { SystemStatus, CachedModelInfo, LoadedModelInfo, DownloadProgress } from '../api/types';
import { api } from '../api/client';

export interface DownloadingModel {
  repoId: string;
  progress: DownloadProgress | null;
  error: string | null;
}

interface SystemState {
  status: SystemStatus | null;
  cachedModels: CachedModelInfo[];
  loadedModels: LoadedModelInfo[];
  downloadingModels: Map<string, DownloadingModel>;
  isLoading: boolean;
  error: string | null;

  fetchStatus: () => Promise<void>;
  fetchModels: () => Promise<void>;
  fetchLoadedModels: () => Promise<void>;
  deleteModel: (repoId: string) => Promise<boolean>;
  unloadModel: (key: string) => Promise<boolean>;
  startDownload: (repoId: string) => Promise<void>;
  isDownloading: (repoId: string) => boolean;
  getDownloadProgress: (repoId: string) => DownloadingModel | undefined;
}

export const useSystemStore = create<SystemState>((set, get) => ({
  status: null,
  cachedModels: [],
  loadedModels: [],
  downloadingModels: new Map(),
  isLoading: false,
  error: null,

  fetchStatus: async () => {
    try {
      const response = await api.getSystemStatus();
      set({ status: response.status, error: null });
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },

  fetchModels: async () => {
    set({ isLoading: true });
    try {
      const response = await api.getCachedModels();
      set({ cachedModels: response.models, isLoading: false, error: null });
    } catch (e) {
      set({ isLoading: false, error: (e as Error).message });
    }
  },

  fetchLoadedModels: async () => {
    try {
      const loadedModels = await api.getLoadedModels();
      set({ loadedModels, error: null });
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },

  deleteModel: async (repoId: string) => {
    try {
      const response = await api.deleteModel(repoId);
      if (response.ok) {
        await get().fetchModels();
        return true;
      }
      return false;
    } catch {
      return false;
    }
  },

  unloadModel: async (key: string) => {
    try {
      const response = await api.unloadModel(key);
      if (response.ok) {
        await get().fetchLoadedModels();
        return true;
      }
      return false;
    } catch {
      return false;
    }
  },

  startDownload: async (repoId: string) => {
    const downloading = new Map(get().downloadingModels);
    downloading.set(repoId, { repoId, progress: null, error: null });
    set({ downloadingModels: downloading });

    try {
      for await (const progress of api.downloadModel(repoId)) {
        const updated = new Map(get().downloadingModels);
        const current = updated.get(repoId);
        if (current) {
          current.progress = progress;
          if (progress.status === 'Failed') {
            current.error = progress.error || 'Download failed';
          }
          set({ downloadingModels: updated });
        }

        if (progress.status === 'Completed' || progress.status === 'Failed') {
          // Remove from downloading after a short delay
          setTimeout(() => {
            const final = new Map(get().downloadingModels);
            final.delete(repoId);
            set({ downloadingModels: final });
          }, progress.status === 'Completed' ? 2000 : 5000);

          if (progress.status === 'Completed') {
            await get().fetchModels();
          }
          break;
        }
      }
    } catch (e) {
      const updated = new Map(get().downloadingModels);
      const current = updated.get(repoId);
      if (current) {
        current.error = (e as Error).message;
        set({ downloadingModels: updated });
      }
      // Remove after delay
      setTimeout(() => {
        const final = new Map(get().downloadingModels);
        final.delete(repoId);
        set({ downloadingModels: final });
      }, 5000);
    }
  },

  isDownloading: (repoId: string) => {
    return get().downloadingModels.has(repoId);
  },

  getDownloadProgress: (repoId: string) => {
    return get().downloadingModels.get(repoId);
  },
}));

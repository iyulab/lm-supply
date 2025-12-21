import type {
  SystemStatusResponse,
  CachedModelsResponse,
  LoadedModelInfo,
  ChatRequest,
  EmbedRequest,
  EmbedResponse,
  RerankRequest,
  RerankResponse,
  SynthesizeRequest,
  TranscribeResponse,
  ModelCheckResult,
  DownloadProgress,
  CaptionResponse,
  VqaResponse,
  OcrResponse,
  OcrLanguage,
  DetectResponse,
  DetectionLabel,
  SegmentResponse,
  SegmentLabel,
  TranslateRequest,
  TranslateResponse,
  TranslateLanguage,
  ModelRegistry,
  ModelTypeInfo,
  ImageGenerationRequest,
  ImageGenerationExtendedResponse,
  ImageModelInfo,
} from './types';

const API_BASE = '/api';
const V1_BASE = '/v1';

// Direct backend URL for SSE endpoints (bypasses Vite proxy to avoid chunked encoding issues)
const getBackendUrl = () => {
  // In development, connect directly to backend for SSE
  if (import.meta.env.DEV) {
    return 'http://localhost:5000';
  }
  // In production, use relative URLs (same origin)
  return '';
};

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ message: response.statusText }));
    throw new Error(error.error?.message || error.message || error.detail || 'Request failed');
  }

  return response.json();
}

export const api = {
  // ============================================================================
  // System Endpoints
  // ============================================================================

  getSystemStatus: () => fetchJson<SystemStatusResponse>(`${API_BASE}/system/status`),

  // ============================================================================
  // Cache Management Endpoints
  // ============================================================================

  getCachedModels: () => fetchJson<CachedModelsResponse>(`${API_BASE}/cache/models`),

  getLoadedModels: () => fetchJson<LoadedModelInfo[]>(`${API_BASE}/cache/loaded`),

  deleteModel: (repoId: string) =>
    fetch(`${API_BASE}/cache/models/${encodeURIComponent(repoId)}`, { method: 'DELETE' }),

  // Note: Backend doesn't have a dedicated unload endpoint.
  // Models are unloaded automatically when deleted or when memory pressure occurs.
  unloadModel: async (_key: string) => {
    return new Response(null, { status: 501, statusText: 'Not Implemented' });
  },

  // ============================================================================
  // Download Management Endpoints
  // ============================================================================

  checkModel: (repoId: string) =>
    fetchJson<ModelCheckResult>(`${API_BASE}/download/check`, {
      method: 'POST',
      body: JSON.stringify({ repoId }),
    }),

  downloadModel: async function* (repoId: string): AsyncGenerator<DownloadProgress> {
    // Use direct backend URL for SSE to bypass Vite proxy
    const backendUrl = getBackendUrl();
    const response = await fetch(`${backendUrl}${API_BASE}/download/model`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ repoId }),
    });

    if (!response.ok) {
      throw new Error('Download request failed');
    }

    const reader = response.body?.getReader();
    if (!reader) throw new Error('No response body');

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6).trim();
          if (!data) continue;
          try {
            const parsed = JSON.parse(data) as DownloadProgress;
            yield parsed;
            if (parsed.status === 'Completed' || parsed.status === 'Failed') return;
          } catch {
            // Skip non-JSON lines
          }
        }
      }
    }
  },

  // ============================================================================
  // Chat Endpoints (OpenAI Compatible)
  // ============================================================================

  chatStream: async function* (request: ChatRequest): AsyncGenerator<string> {
    const response = await fetch(`${V1_BASE}/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: request.modelId,
        messages: request.messages.map(m => ({ role: m.role, content: m.content })),
        stream: true,
        max_tokens: request.options?.maxTokens,
        temperature: request.options?.temperature,
        top_p: request.options?.topP,
      }),
      signal: request.signal,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ message: 'Chat request failed' }));
      throw new Error(error.error?.message || error.message || 'Chat request failed');
    }

    const reader = response.body?.getReader();
    if (!reader) throw new Error('No response body');

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') return;
          try {
            const parsed = JSON.parse(data);
            const content = parsed.choices?.[0]?.delta?.content;
            if (content) yield content;
          } catch {
            // Skip non-JSON lines
          }
        }
      }
    }
  },

  chatComplete: async (request: ChatRequest) => {
    const response = await fetchJson<{
      id: string;
      model: string;
      choices: Array<{ message: { role: string; content: string } }>;
      usage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
    }>(`${V1_BASE}/chat/completions`, {
      method: 'POST',
      body: JSON.stringify({
        model: request.modelId,
        messages: request.messages.map(m => ({ role: m.role, content: m.content })),
        stream: false,
        max_tokens: request.options?.maxTokens,
        temperature: request.options?.temperature,
        top_p: request.options?.topP,
      }),
    });
    return {
      modelId: response.model,
      response: response.choices[0]?.message?.content ?? '',
      usage: response.usage,
    };
  },

  // ============================================================================
  // Embed Endpoints (OpenAI Compatible)
  // ============================================================================

  embed: (request: EmbedRequest) =>
    fetchJson<EmbedResponse>(`${V1_BASE}/embeddings`, {
      method: 'POST',
      body: JSON.stringify({
        model: request.modelId,
        input: request.texts,
      }),
    }),

  // ============================================================================
  // Rerank Endpoints (Cohere Compatible)
  // ============================================================================

  rerank: (request: RerankRequest) =>
    fetchJson<RerankResponse>(`${V1_BASE}/rerank`, {
      method: 'POST',
      body: JSON.stringify({
        model: request.modelId,
        query: request.query,
        documents: request.documents,
        top_n: request.topN,
        return_documents: request.returnDocuments ?? true,
      }),
    }),

  // ============================================================================
  // Synthesize Endpoints (OpenAI Compatible)
  // ============================================================================

  synthesize: async (request: SynthesizeRequest): Promise<Blob> => {
    const response = await fetch(`${V1_BASE}/audio/speech`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: request.modelId,
        input: request.text,
        voice: request.voice,
        response_format: request.responseFormat ?? 'wav',
        speed: request.speed,
      }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ message: 'Synthesis failed' }));
      throw new Error(error.error?.message || error.message || 'Synthesis failed');
    }

    return response.blob();
  },

  // ============================================================================
  // Transcribe Endpoints (OpenAI Compatible)
  // ============================================================================

  transcribe: async (file: File, modelId: string = 'default', responseFormat: string = 'json'): Promise<TranscribeResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', modelId);
    formData.append('response_format', responseFormat);

    const response = await fetch(`${V1_BASE}/audio/transcriptions`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ message: 'Transcription failed' }));
      throw new Error(error.error?.message || error.message || 'Transcription failed');
    }

    return response.json();
  },

  // ============================================================================
  // Caption Endpoints
  // ============================================================================

  caption: async (file: File, modelId: string = 'default'): Promise<CaptionResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', modelId);

    const response = await fetch(`${V1_BASE}/images/caption`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ message: 'Captioning failed' }));
      throw new Error(error.error?.message || error.message || 'Captioning failed');
    }

    return response.json();
  },

  vqa: async (file: File, question: string, modelId: string = 'default'): Promise<VqaResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('question', question);
    formData.append('model', modelId);

    const response = await fetch(`${V1_BASE}/images/vqa`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ message: 'VQA failed' }));
      throw new Error(error.error?.message || error.message || 'VQA failed');
    }

    return response.json();
  },

  // ============================================================================
  // OCR Endpoints
  // ============================================================================

  ocr: async (file: File, language: string = 'en'): Promise<OcrResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('language', language);

    const response = await fetch(`${V1_BASE}/images/ocr`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ message: 'OCR failed' }));
      throw new Error(error.error?.message || error.message || 'OCR failed');
    }

    return response.json();
  },

  getOcrLanguages: async (): Promise<OcrLanguage[]> => {
    const response = await fetchJson<{ languages: OcrLanguage[] }>(`${V1_BASE}/images/ocr/languages`);
    return response.languages;
  },

  // ============================================================================
  // Detect Endpoints
  // ============================================================================

  detect: async (file: File, modelId: string = 'default', threshold: number = 0.5): Promise<DetectResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', modelId);
    formData.append('threshold', threshold.toString());

    const response = await fetch(`${V1_BASE}/images/detect`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ message: 'Detection failed' }));
      throw new Error(error.error?.message || error.message || 'Detection failed');
    }

    return response.json();
  },

  getDetectionLabels: async (): Promise<DetectionLabel[]> => {
    const response = await fetchJson<{ labels: DetectionLabel[] }>(`${V1_BASE}/images/detect/labels`);
    return response.labels;
  },

  // ============================================================================
  // Segment Endpoints
  // ============================================================================

  segment: async (file: File, modelId: string = 'default', includeMask: boolean = false): Promise<SegmentResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', modelId);
    formData.append('include_mask', includeMask.toString());

    const response = await fetch(`${V1_BASE}/images/segment`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ message: 'Segmentation failed' }));
      throw new Error(error.error?.message || error.message || 'Segmentation failed');
    }

    return response.json();
  },

  getSegmentLabels: async (): Promise<SegmentLabel[]> => {
    const response = await fetchJson<{ labels: SegmentLabel[] }>(`${V1_BASE}/images/segment/labels`);
    return response.labels;
  },

  // ============================================================================
  // Translate Endpoints
  // ============================================================================

  translate: (request: TranslateRequest) =>
    fetchJson<TranslateResponse>(`${V1_BASE}/translate`, {
      method: 'POST',
      body: JSON.stringify({
        model: request.modelId,
        input: request.text ?? request.texts,
        source_language: request.sourceLanguage,
        target_language: request.targetLanguage,
      }),
    }),

  getTranslateLanguages: async (): Promise<TranslateLanguage[]> => {
    const response = await fetchJson<{ languages: TranslateLanguage[] }>(`${V1_BASE}/translate/languages`);
    return response.languages;
  },

  // ============================================================================
  // Model Registry Endpoints
  // ============================================================================

  getModelRegistry: () => fetchJson<ModelRegistry>(`${API_BASE}/registry/models`),

  getModelsByType: (type: string) => fetchJson<ModelTypeInfo>(`${API_BASE}/registry/models/${encodeURIComponent(type)}`),

  // ============================================================================
  // Image Generation Endpoints
  // ============================================================================

  generateImage: (request: ImageGenerationRequest) =>
    fetchJson<ImageGenerationExtendedResponse>(`${V1_BASE}/images/generate`, {
      method: 'POST',
      body: JSON.stringify({
        prompt: request.prompt,
        model: request.model,
        size: request.size,
        steps: request.steps,
        guidance_scale: request.guidance_scale,
        seed: request.seed,
        negative_prompt: request.negative_prompt,
      }),
    }),

  getImageModels: async (): Promise<ImageModelInfo[]> => {
    const response = await fetchJson<{ models: ImageModelInfo[] }>(`${V1_BASE}/images/models`);
    return response.models;
  },
};

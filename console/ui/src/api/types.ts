// ============================================================================
// System Types
// ============================================================================

export interface SystemStatus {
  engineReady: boolean;
  gpuAvailable: boolean;
  gpuProvider: string;
  gpuName?: string;
  cpuUsage: number;
  ramUsageMB: number;
  ramTotalMB: number;
  ramUsagePercent: number;
  vramUsageMB?: number;
  vramTotalMB?: number;
  vramUsagePercent?: number;
  processMemoryMB: number;
  timestamp: string;
}

export interface SystemStatusResponse {
  status: SystemStatus;
  loadedModels: number;
  models: LoadedModelInfo[];
}

// ============================================================================
// Model/Cache Types
// ============================================================================

export type ModelType =
  | 'Generator'
  | 'Embedder'
  | 'Reranker'
  | 'Transcriber'
  | 'Synthesizer'
  | 'Translator'
  | 'Captioner'
  | 'Ocr'
  | 'Detector'
  | 'Segmenter'
  | 'Unknown';

export interface CachedModelInfo {
  repoId: string;
  localPath: string;
  sizeBytes: number;
  sizeMB: number;
  fileCount: number;
  detectedType: ModelType;
  lastModified: string;
  files: string[];
}

export interface LoadedModelInfo {
  modelId: string;
  modelType: string;
  loadedAt: string;
  lastUsedAt: string;
}

export interface CachedModelsResponse {
  models: CachedModelInfo[];
  totalCount: number;
  totalSizeMB: number;
}

export interface ModelCheckResult {
  exists: boolean;
  repoId: string;
  detectedType: ModelType;
  fileCount: number;
  totalSizeBytes: number;
  totalSizeMB: number;
  files: string[];
  error?: string;
}

export interface DownloadProgress {
  fileName: string;
  bytesDownloaded: number;
  totalBytes: number;
  percentComplete: number;
  status?: string;
  error?: string;
}

// ============================================================================
// Chat Types (OpenAI Compatible)
// ============================================================================

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface ChatOptions {
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  repetitionPenalty?: number;
  stopSequences?: string[];
}

export interface ChatRequest {
  modelId: string;
  messages: ChatMessage[];
  options?: ChatOptions;
}

// ============================================================================
// Embed Types (OpenAI Compatible)
// ============================================================================

export interface EmbedRequest {
  modelId: string;
  texts: string[];
  normalize?: boolean;
}

export interface EmbeddingData {
  object: string;
  index: number;
  embedding: number[];
}

export interface EmbeddingUsage {
  prompt_tokens: number;
  total_tokens: number;
}

export interface EmbedResponse {
  object: string;
  data: EmbeddingData[];
  model: string;
  usage: EmbeddingUsage;
}

// ============================================================================
// Rerank Types (Cohere Compatible)
// ============================================================================

export interface RerankRequest {
  modelId: string;
  query: string;
  documents: string[];
  topN?: number;
  returnDocuments?: boolean;
}

export interface RerankResultDocument {
  text: string;
}

export interface RerankResult {
  index: number;
  relevance_score: number;
  document?: RerankResultDocument;
}

export interface RerankResponse {
  id: string;
  model: string;
  results: RerankResult[];
}

// ============================================================================
// Synthesize Types (OpenAI Compatible)
// ============================================================================

export interface SynthesizeRequest {
  modelId: string;
  text: string;
  voice?: string;
  responseFormat?: string;
  speed?: number;
}

// Response is binary audio (Blob)

// ============================================================================
// Transcribe Types (OpenAI Compatible)
// ============================================================================

export interface TranscriptionSegment {
  id: number;
  start: number;
  end: number;
  text: string;
}

export interface TranscribeResponse {
  text: string;
  // Verbose format additional fields
  task?: string;
  language?: string;
  duration?: number;
  segments?: TranscriptionSegment[];
}

// ============================================================================
// Caption Types
// ============================================================================

export interface CaptionResponse {
  id: string;
  model: string;
  caption: string;
  confidence?: number;
  alternatives?: string[];
}

export interface VqaResponse {
  id: string;
  model: string;
  question: string;
  answer: string;
  confidence?: number;
}

// ============================================================================
// OCR Types
// ============================================================================

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface OcrBlock {
  text: string;
  confidence: number;
  boundingBox?: BoundingBox;
}

export interface OcrResponse {
  id: string;
  model: string;
  text: string;
  blocks?: OcrBlock[];
}

export interface OcrLanguage {
  code: string;
  name?: string;
}

// ============================================================================
// Detect Types
// ============================================================================

export interface DetectedObject {
  label: string;
  confidence: number;
  bounding_box: BoundingBox;
}

export interface DetectResponse {
  id: string;
  model: string;
  objects: DetectedObject[];
}

export interface DetectionLabel {
  id: number;
  name: string;
}

// ============================================================================
// Segment Types
// ============================================================================

export interface Segment {
  id: number;
  label?: string;
  score?: number;
  bounding_box?: BoundingBox;
  maskBase64?: string;
}

export interface SegmentResponse {
  id: string;
  model: string;
  segments: Segment[];
  maskBase64?: string;
}

export interface SegmentLabel {
  id: number;
  name: string;
}

// ============================================================================
// Translate Types
// ============================================================================

export interface TranslateRequest {
  modelId: string;
  text?: string;
  texts?: string[];
  sourceLanguage?: string;
  targetLanguage?: string;
}

export interface TranslationResult {
  index: number;
  source_text: string;
  translated_text: string;
  source_language?: string;
  target_language?: string;
}

export interface TranslateResponse {
  id: string;
  model: string;
  translations: TranslationResult[];
}

export interface TranslateLanguage {
  id: string;
  alias: string;
  source: string;
  target: string;
}

// ============================================================================
// Model Registry Types
// ============================================================================

export interface ModelAliasInfo {
  alias: string;
  repoId: string;
  description: string;
  isCached: boolean;
}

export interface ModelTypeInfo {
  type: string;
  displayName: string;
  description: string;
  models: ModelAliasInfo[];
}

export interface ModelRegistry {
  modelTypes: ModelTypeInfo[];
}

// ============================================================================
// Image Generation Types
// ============================================================================

export interface ImageGenerationRequest {
  prompt: string;
  model?: string;
  n?: number;
  size?: string;
  response_format?: string;
  steps?: number;
  guidance_scale?: number;
  seed?: number;
  negative_prompt?: string;
}

export interface GeneratedImageData {
  b64_json?: string;
  url?: string;
  revised_prompt?: string;
}

export interface ImageGenerationResponse {
  created: number;
  data: GeneratedImageData[];
}

export interface GeneratedImageExtendedData {
  b64_json?: string;
  width: number;
  height: number;
  seed: number;
  steps: number;
  prompt?: string;
}

export interface ImageGenerationExtendedResponse {
  id: string;
  model: string;
  created: number;
  data: GeneratedImageExtendedData[];
  generation_time_ms: number;
}

export interface ImageModelInfo {
  id: string;
  repo_id: string;
  recommended_steps: number;
  recommended_guidance_scale: number;
}

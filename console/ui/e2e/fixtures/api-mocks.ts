import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Page, Route } from '@playwright/test';
import type { CaptionResponse, OcrResponse } from '../../src/api/types';

/**
 * API mock helpers for Playwright E2E tests.
 * Uses page.route() to intercept API calls and return mock responses.
 * The live backend handles ModelSelector API calls; only inference endpoints are mocked.
 */

// ============================================================================
// JSON Response Mocks
// ============================================================================

/** Mock a JSON API endpoint */
export async function mockJsonEndpoint(
  page: Page,
  urlPattern: string | RegExp,
  response: object,
  options?: { status?: number; delay?: number }
) {
  await page.route(urlPattern, async (route: Route) => {
    if (options?.delay) {
      await new Promise(resolve => setTimeout(resolve, options.delay));
    }
    await route.fulfill({
      status: options?.status ?? 200,
      contentType: 'application/json',
      body: JSON.stringify(response),
    });
  });
}

/** Mock a JSON API endpoint to return an error */
export async function mockJsonError(
  page: Page,
  urlPattern: string | RegExp,
  message: string,
  status: number = 500
) {
  await page.route(urlPattern, async (route: Route) => {
    await route.fulfill({
      status,
      contentType: 'application/json',
      body: JSON.stringify({ error: { message } }),
    });
  });
}

// ============================================================================
// Blob Response Mocks (for audio/image binary endpoints)
// ============================================================================

/** Mock a binary blob endpoint (e.g., synthesize audio) */
export async function mockBlobEndpoint(
  page: Page,
  urlPattern: string | RegExp,
  contentType: string,
  body: Buffer
) {
  await page.route(urlPattern, async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType,
      body,
    });
  });
}

// ============================================================================
// SSE Stream Mocks (for chat streaming)
// ============================================================================

/** Mock an SSE streaming endpoint (OpenAI chat completions format) */
export async function mockChatStream(
  page: Page,
  urlPattern: string | RegExp,
  tokens: string[],
  options?: { delay?: number }
) {
  await page.route(urlPattern, async (route: Route) => {
    const tokenDelay = options?.delay ?? 10;
    const lines: string[] = [];

    for (const token of tokens) {
      lines.push(`data: ${JSON.stringify({
        id: 'chatcmpl-mock',
        object: 'chat.completion.chunk',
        choices: [{ index: 0, delta: { content: token }, finish_reason: null }],
      })}\n\n`);
    }

    // Final done signal
    lines.push('data: [DONE]\n\n');

    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      body: lines.join(''),
    });
  });
}

// ============================================================================
// Mock Data Factories
// ============================================================================

/** Generate a mock embedding vector */
export function mockEmbeddingVector(dimensions: number = 384): number[] {
  return Array.from({ length: dimensions }, (_, i) =>
    Math.sin(i * 0.1) * 0.5 + Math.cos(i * 0.3) * 0.3
  );
}

/** Generate mock embed response */
export function mockEmbedResponse(texts: string[], dimensions: number = 384) {
  return {
    object: 'list',
    data: texts.map((_, index) => ({
      object: 'embedding',
      index,
      embedding: mockEmbeddingVector(dimensions),
    })),
    model: 'mock-embedder',
    usage: { prompt_tokens: texts.length * 5, total_tokens: texts.length * 5 },
  };
}

/** Generate mock rerank response */
export function mockRerankResponse(documents: string[], query: string) {
  return {
    id: 'rerank-mock',
    model: 'mock-reranker',
    results: documents.map((text, i) => ({
      index: i,
      relevance_score: 0.95 - i * 0.15,
      document: { text },
    })).sort((a, b) => b.relevance_score - a.relevance_score),
  };
}

/** Generate mock transcribe response */
export function mockTranscribeResponse() {
  return {
    text: 'Hello, this is a test transcription of the audio file.',
    task: 'transcribe',
    language: 'en',
    duration: 3.45,
    segments: [
      { id: 0, start: 0.0, end: 1.5, text: 'Hello, this is a test' },
      { id: 1, start: 1.5, end: 3.45, text: 'transcription of the audio file.' },
    ],
  };
}

/** Generate mock detect response */
export function mockDetectResponse() {
  return {
    id: 'detect-mock',
    model: 'mock-detector',
    objects: [
      { label: 'person', confidence: 0.95, bounding_box: { x: 10, y: 20, width: 100, height: 200 } },
      { label: 'person', confidence: 0.88, bounding_box: { x: 150, y: 30, width: 90, height: 180 } },
      { label: 'car', confidence: 0.82, bounding_box: { x: 300, y: 150, width: 200, height: 100 } },
    ],
  };
}

/** Generate mock segment response */
export function mockSegmentResponse() {
  return {
    id: 'segment-mock',
    model: 'mock-segmenter',
    segments: [
      { id: 0, label: 'sky', score: 0.42 },
      { id: 1, label: 'building', score: 0.28 },
      { id: 2, label: 'road', score: 0.15 },
      { id: 3, label: 'tree', score: 0.10 },
      { id: 4, label: 'car', score: 0.05 },
    ],
  };
}

/** Generate mock translate response */
export function mockTranslateResponse(text: string) {
  return {
    id: 'translate-mock',
    model: 'mock-translator',
    translations: [{
      index: 0,
      source_text: text,
      translated_text: 'Translated: ' + text,
      source_language: 'en',
      target_language: 'ko',
    }],
  };
}

/** Caption response in the shape the running host answers with (e2e/fixtures/caption-response.real.json, captured from POST /v1/images/caption). */
export function mockCaptionResponse(alternatives: { caption: string; confidence: number }[] = []) {
  const real = JSON.parse(readFileSync(join(dirname(fileURLToPath(import.meta.url)), 'caption-response.real.json'), 'utf-8')) as CaptionResponse;
  const extra = alternatives.map((alt, i) => ({ ...real.choices[0], index: i + 1, caption: alt.caption, confidence: alt.confidence }));
  return { ...real, choices: [...real.choices, ...extra] };
}

/** Generate mock VQA response */
export function mockVqaResponse(question: string) {
  return {
    id: 'vqa-mock',
    model: 'mock-captioner',
    question,
    answer: 'The image shows a red pixel.',
    confidence: 0.92,
  };
}

/**
 * OCR response in the shape the running host answers with: a real response (e2e/fixtures/ocr-response.real.json, captured from
 * POST /v1/images/ocr) rather than a hand-written one, so a UI that reads another shape fails here instead of in front of a user.
 */
export function mockOcrResponse() {
  return JSON.parse(readFileSync(join(dirname(fileURLToPath(import.meta.url)), 'ocr-response.real.json'), 'utf-8')) as OcrResponse;
}

/** The same real response with the recognized lines replaced — for tests that need other text (another language, an empty page). */
export function mockOcrResponseWithLines(lines: { text: string; confidence: number }[]): OcrResponse {
  const real = mockOcrResponse();
  const template = real.pages[0].blocks[0];
  const blocks = lines.map((line, index) => ({
    ...template,
    text: line.text,
    confidence: line.confidence,
    bounding_box: { ...template.bounding_box, y: template.bounding_box.y + index * 60 },
    lines: [{ text: line.text, confidence: line.confidence, bounding_box: { ...template.bounding_box, y: template.bounding_box.y + index * 60 } }],
  }));
  return { ...real, full_text: lines.map((line) => line.text).join('\n'), pages: [{ ...real.pages[0], blocks }] };
}

/** Generate a minimal WAV buffer (for synthesize mock) */
export function mockWavBuffer(): Buffer {
  // Minimal WAV: RIFF header + fmt chunk + data chunk with 1 second of silence
  const sampleRate = 8000;
  const numSamples = sampleRate; // 1 second
  const dataSize = numSamples * 2; // 16-bit
  const buffer = Buffer.alloc(44 + dataSize);

  // RIFF header
  buffer.write('RIFF', 0);
  buffer.writeUInt32LE(36 + dataSize, 4);
  buffer.write('WAVE', 8);

  // fmt chunk
  buffer.write('fmt ', 12);
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20);  // PCM
  buffer.writeUInt16LE(1, 22);  // Mono
  buffer.writeUInt32LE(sampleRate, 24);
  buffer.writeUInt32LE(sampleRate * 2, 28);
  buffer.writeUInt16LE(2, 32);
  buffer.writeUInt16LE(16, 34);

  // data chunk
  buffer.write('data', 36);
  buffer.writeUInt32LE(dataSize, 40);
  // Data is all zeros (silence)

  return buffer;
}

// ============================================================================
// Page-Level Mock Helpers (mock all APIs for a given page)
// ============================================================================

/** Mock all Dashboard page API endpoints */
export async function mockDashboardApis(page: Page, options?: {
  cachedModels?: object[];
  loadedModels?: object[];
}) {
  const cached = options?.cachedModels ?? [];
  const loaded = options?.loadedModels ?? [];

  await page.route('**/api/cache/models', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ models: cached }),
    });
  });

  await page.route('**/api/cache/loaded', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(loaded),
    });
  });
}

/** Mock all Models page API endpoints */
export async function mockModelsPageApis(page: Page, options?: {
  cachedModels?: object[];
  loadedModels?: object[];
  registryTypes?: object[];
}) {
  const cached = options?.cachedModels ?? [];
  const loaded = options?.loadedModels ?? [];
  const registry = options?.registryTypes ?? [];

  await page.route('**/api/cache/models', async (route: Route) => {
    if (route.request().method() === 'GET') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ models: cached }),
      });
    } else if (route.request().method() === 'DELETE') {
      await route.fulfill({ status: 200, body: '{}' });
    } else {
      await route.continue();
    }
  });

  await page.route('**/api/cache/loaded', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(loaded),
    });
  });

  await page.route('**/api/registry/models', async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ modelTypes: registry }),
    });
  });
}

// ============================================================================
// Image Generation Mocks
// ============================================================================

/** Generate mock image generation response */
export function mockImageGenerationResponse() {
  // 1x1 red pixel PNG as base64
  const pngBase64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==';
  return {
    id: 'imggen-mock',
    model: 'mock-generator',
    created: Date.now(),
    data: [{
      b64_json: pngBase64,
      width: 512,
      height: 512,
      seed: 42,
      steps: 4,
      prompt: 'A test image',
    }],
    generation_time_ms: 1234,
  };
}

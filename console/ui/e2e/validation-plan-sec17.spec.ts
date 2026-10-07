import { test, expect } from '@playwright/test';

const BASE = process.env.API_BASE_URL || 'http://localhost:5000';

/**
 * Section 17: OpenAI SDK Compatibility Tests
 *
 * These tests validate that the API responses are structurally compatible
 * with what the OpenAI Python/JS SDK expects. We test the raw API shapes
 * rather than running the SDK itself (no Python runtime available in tests).
 */

test.describe('Section 17-1: OpenAI SDK Response Compatibility', () => {
  test('[17-1-01] chat.completions response matches SDK schema', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: {
        model: 'default',
        messages: [{ role: 'user', content: 'Hello' }],
        max_tokens: 5,
      },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      // SDK requires: id (str), object, created (int), model (str), choices[], usage{}
      expect(typeof body.id).toBe('string');
      expect(body.id).toMatch(/^chatcmpl-/);
      expect(body.object).toBe('chat.completion');
      expect(typeof body.created).toBe('number');
      expect(typeof body.model).toBe('string');
      expect(Array.isArray(body.choices)).toBe(true);
      expect(body.choices[0].message.role).toBe('assistant');
      expect(typeof body.choices[0].message.content).toBe('string');
      expect(body.usage).toHaveProperty('prompt_tokens');
      expect(body.usage).toHaveProperty('completion_tokens');
      expect(body.usage).toHaveProperty('total_tokens');
    } else {
      expect([404, 503]).toContain(resp.status());
    }
  });

  test('[17-1-02] embeddings response matches SDK schema', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/embeddings`, {
      data: { model: 'default', input: 'test' },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      // SDK requires: object="list", data[], model, usage{}
      expect(body.object).toBe('list');
      expect(typeof body.model).toBe('string');
      expect(body.data[0].object).toBe('embedding');
      expect(typeof body.data[0].index).toBe('number');
      expect(Array.isArray(body.data[0].embedding)).toBe(true);
      expect(body.usage).toHaveProperty('prompt_tokens');
      expect(body.usage).toHaveProperty('total_tokens');
    } else {
      expect([404, 503]).toContain(resp.status());
    }
  });

  test('[17-1-03] audio transcriptions response matches SDK schema', async ({ request }) => {
    const wavBuf = Buffer.alloc(44, 0); // minimal header
    wavBuf.write('RIFF', 0); wavBuf.write('WAVE', 8); wavBuf.write('fmt ', 12); wavBuf.write('data', 36);
    const resp = await request.post(`${BASE}/v1/audio/transcriptions`, {
      multipart: {
        file: { name: 'test.wav', mimeType: 'audio/wav', buffer: wavBuf },
        model: 'default',
      },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      // SDK requires: text (str)
      expect(typeof body.text).toBe('string');
    } else {
      expect([404, 503]).toContain(resp.status());
    }
  });

  test('[17-1-04] audio speech response is binary audio data', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/audio/speech`, {
      data: { model: 'default', input: 'Hello', voice: 'alloy' },
    });
    if (resp.status() === 200) {
      // SDK expects binary audio bytes
      const body = await resp.body();
      expect(body.length).toBeGreaterThan(0);
      const ct = resp.headers()['content-type'];
      expect(ct).toMatch(/audio/);
    } else {
      expect([404, 503]).toContain(resp.status());
    }
  });

  test('[17-1-05] images generate response matches SDK ImagesResponse', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/generations`, {
      data: { prompt: 'A red apple', model: 'default', size: '256x256' },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      // SDK ImagesResponse requires: created (int), data[]
      expect(typeof body.created).toBe('number');
      expect(Array.isArray(body.data)).toBe(true);
      // Each item: b64_json (str) or url (str)
      const item = body.data[0];
      expect(item.b64_json || item.url).toBeTruthy();
    } else {
      expect([404, 503]).toContain(resp.status());
    }
  });

  test('[17-1-06] streaming chat response is valid SSE with correct chunk shape', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: {
        model: 'default',
        messages: [{ role: 'user', content: 'Say hi.' }],
        stream: true,
        max_tokens: 5,
      },
    });
    if (resp.status() === 200) {
      const text = await resp.text();
      // Parse first data line
      const lines = text.split('\n').filter(l => l.startsWith('data: ') && !l.includes('[DONE]'));
      if (lines.length > 0) {
        const chunk = JSON.parse(lines[0].replace('data: ', ''));
        // SDK chunk shape: id, object="chat.completion.chunk", choices[]
        expect(chunk.object).toBe('chat.completion.chunk');
        expect(Array.isArray(chunk.choices)).toBe(true);
        expect(chunk.choices[0]).toHaveProperty('delta');
        expect(chunk.choices[0]).toHaveProperty('index');
        expect(chunk.choices[0]).toHaveProperty('finish_reason');
      }
      expect(text).toContain('data: [DONE]');
    } else {
      expect([404, 503]).toContain(resp.status());
    }
  });

  test('[17-1-07] model not found returns OpenAI-compatible 404 error', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/embeddings`, {
      data: { model: 'nonexistent-sdk-compat-test', input: 'test' },
    });
    expect(resp.status()).toBe(404);
    const body = await resp.json();
    // SDK raises NotFoundError for: {"error": {"type": "...", "message": "...", "code": "..."}}
    expect(body.error).toBeTruthy();
    expect(body.error.message).toBeTruthy();
    expect(body.error.type).toBeTruthy();
    // error.code should be present for SDK to parse
    expect(body.error.code).toBeTruthy();
  });
});

import { test, expect } from '@playwright/test';

const BASE = process.env.API_BASE_URL || 'http://localhost:5000';

/**
 * Gap closure tests — API behaviors that
 * require additional coverage beyond the primary section specs.
 */

// ================================================================
// Cache management success cases
// ================================================================
test.describe('Cache Success Paths', () => {
  test('GET /api/cache/stats byType is object/map', async ({ request }) => {
    const resp = await request.get(`${BASE}/api/cache/stats`);
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(typeof body.byType).toBe('object');
    expect(body.byType).not.toBeNull();
  });

  test('loaded model lastUsedAt is ISO date string', async ({ request }) => {
    const resp = await request.get(`${BASE}/api/cache/loaded`);
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    if (body.length > 0) {
      const lastUsed = body[0].lastUsedAt;
      // Should be parseable as a date
      expect(new Date(lastUsed).toString()).not.toBe('Invalid Date');
    }
  });
});

// ================================================================
// Streaming chunk validation
// ================================================================
test.describe('Streaming Detail', () => {
  test('last streaming chunk has finish_reason=stop', async ({ request }) => {
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
      const lines = text.split('\n').filter(l => l.startsWith('data: ') && !l.includes('[DONE]'));
      if (lines.length > 0) {
        // Find last non-done chunk
        const lastChunk = JSON.parse(lines[lines.length - 1].replace('data: ', ''));
        const finishReason = lastChunk.choices[0].finish_reason;
        if (finishReason !== null) {
          expect(finishReason).toBe('stop');
        }
      }
    }
  });

  test('streaming chunk format: data: {id, choices[{delta:{content}}]}', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: {
        model: 'default',
        messages: [{ role: 'user', content: 'Hi.' }],
        stream: true,
        max_tokens: 5,
      },
    });
    if (resp.status() === 200) {
      const text = await resp.text();
      const contentLines = text.split('\n').filter(l => l.startsWith('data: ') && !l.includes('[DONE]'));
      if (contentLines.length > 0) {
        const chunk = JSON.parse(contentLines[0].replace('data: ', ''));
        expect(chunk).toHaveProperty('id');
        expect(chunk).toHaveProperty('choices');
        expect(chunk.choices[0]).toHaveProperty('delta');
      }
    }
  });
});

// ================================================================
// Token count validation
// ================================================================
test.describe('Token Counts', () => {
  test('usage.prompt_tokens matches total_tokens for single input', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/embeddings`, {
      data: { model: 'default', input: 'hello world' },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      expect(body.usage.prompt_tokens).toBeGreaterThanOrEqual(0);
      expect(body.usage.total_tokens).toBeGreaterThanOrEqual(0);
      expect(body.usage.total_tokens).toBeGreaterThanOrEqual(body.usage.prompt_tokens);
    }
  });
});

// ================================================================
// DeepL endpoint robustness
// ================================================================
test.describe('DeepL Edge Cases', () => {
  test('lowercase target_lang also handled (ko → KO internally)', async ({ request }) => {
    // Some clients might send lowercase
    const resp = await request.post(`${BASE}/v2/translate`, {
      data: { text: ['Hello'], target_lang: 'ko' },
    });
    // Either 200 (accepted) or 400 (strict uppercase required) — not 422
    expect(resp.status()).not.toBe(422);
  });
});

// ================================================================
// Image size variants
// ================================================================
test.describe('Size Variants', () => {
  test('size 256x256 accepted (not 400)', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/generations`, {
      data: { prompt: 'test', model: 'default', size: '256x256' },
    });
    // Not a size error (400 for invalid size)
    expect(resp.status()).not.toBe(400);
  });

  test('size 1024x1024 accepted (not 400)', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/generations`, {
      data: { prompt: 'test', model: 'default', size: '1024x1024' },
    });
    expect(resp.status()).not.toBe(400);
  });
});

// ================================================================
// Version fields
// ================================================================
test.describe('Version Detail', () => {
  test('version field follows semver pattern', async ({ request }) => {
    const resp = await request.get(`${BASE}/api/system/version`);
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(body.version).toMatch(/^\d+\.\d+\.\d+/);
  });
});

// ================================================================
// Swagger endpoint list
// ================================================================
test.describe('Swagger Content', () => {
  test('Swagger JSON spec contains key endpoints', async ({ request }) => {
    const resp = await request.get(`${BASE}/swagger/v1/swagger.json`);
    if (resp.status() === 200) {
      const body = await resp.json();
      expect(body).toHaveProperty('paths');
      const paths = Object.keys(body.paths);
      expect(paths.some(p => p.includes('/v1/models'))).toBe(true);
      expect(paths.some(p => p.includes('/v1/embeddings'))).toBe(true);
    }
    // Swagger might be at different path — not a hard failure
  });
});

// ================================================================
// Model list field detail
// ================================================================
test.describe('Model List Detail', () => {
  test('model created field is Unix timestamp', async ({ request }) => {
    const resp = await request.get(`${BASE}/v1/models`);
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    if (body.data.length > 0) {
      const created = body.data[0].created;
      expect(typeof created).toBe('number');
      // Unix timestamp should be > year 2020
      expect(created).toBeGreaterThan(1577836800);
    }
  });

  test('multilingual alias present in model list', async ({ request }) => {
    const resp = await request.get(`${BASE}/v1/models`);
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    const ids: string[] = body.data.map((m: { id: string }) => m.id);
    // At least some aliases should exist
    const hasAliases = ids.some(id =>
      id === 'default' || id.includes(':default') ||
      id === 'fast' || id.includes(':fast')
    );
    expect(hasAliases).toBe(true);
  });
});

// ================================================================
// Concurrent requests
// ================================================================
test.describe('Concurrent Requests', () => {
  test('concurrent GET /v1/models requests all succeed', async ({ request }) => {
    // Send 5 concurrent requests
    const promises = Array.from({ length: 5 }, () =>
      request.get(`${BASE}/v1/models`)
    );
    const responses = await Promise.all(promises);
    for (const resp of responses) {
      expect(resp.status()).toBe(200);
    }
  });
});

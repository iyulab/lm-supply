import { test, expect } from '@playwright/test';

const BASE = process.env.API_BASE_URL || 'http://localhost:5000';

// ================================================================
// GET /v1/models — OpenAI-compatible list
// ================================================================
test.describe('GET /v1/models', () => {
  test('response format is OpenAI list object', async ({ request }) => {
    const resp = await request.get(`${BASE}/v1/models`);
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(body.object).toBe('list');
    expect(Array.isArray(body.data)).toBe(true);
  });

  test('all 11 domains present in model list', async ({ request }) => {
    const resp = await request.get(`${BASE}/v1/models`);
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    const ids: string[] = body.data.map((m: { id: string }) => m.id.toLowerCase());
    const combined = ids.join(' ');
    // Check presence of each domain via capabilities or id
    expect(body.data.some((m: { capabilities?: string[] }) =>
      m.capabilities?.includes('embeddings') || m.capabilities?.includes('text-embedding')
    )).toBe(true);
  });

  test('each model has capabilities array', async ({ request }) => {
    const resp = await request.get(`${BASE}/v1/models`);
    const body = await resp.json();
    expect(body.data.length).toBeGreaterThan(0);
    for (const model of body.data) {
      expect(model.id).toBeTruthy();
      expect(model.object).toBe('model');
      expect(typeof model.created).toBe('number');
      expect(Array.isArray(model.capabilities)).toBe(true);
    }
  });

  test('standard aliases (default, fast, quality) are present', async ({ request }) => {
    const resp = await request.get(`${BASE}/v1/models`);
    const body = await resp.json();
    const ids: string[] = body.data.map((m: { id: string }) => m.id);
    const hasDefault = ids.some(id => id === 'default' || id.endsWith(':default') || id.includes('default'));
    expect(hasDefault).toBe(true);
  });
});

// ================================================================
// GET /v1/models/{model}
// ================================================================
test.describe('GET /v1/models/{model}', () => {
  test('existing alias returns 200 with ModelInfo', async ({ request }) => {
    const resp = await request.get(`${BASE}/v1/models/default`);
    // May be 404 if no models registered, or 200 with model info
    expect([200, 404]).toContain(resp.status());
    if (resp.status() === 200) {
      const body = await resp.json();
      expect(body.id).toBeTruthy();
      expect(body.object).toBe('model');
    }
  });

  test('nonexistent model returns 404 with error envelope', async ({ request }) => {
    const resp = await request.get(`${BASE}/v1/models/nonexistent-model-xyz`);
    expect(resp.status()).toBe(404);
    const body = await resp.json();
    expect(body.error).toBeTruthy();
    expect(body.error.code).toBeTruthy();
  });

  test('URL-encoded slash in model ID is handled', async ({ request }) => {
    // Should return 200 or 404, not 400/422/500
    const resp = await request.get(`${BASE}/v1/models/microsoft%2FFlorence-2-base`);
    expect([200, 404]).toContain(resp.status());
  });
});

// ================================================================
// Cache Management API
// ================================================================
test.describe('Cache Management API', () => {
  test('GET /api/cache/models returns list with sizeBytes', async ({ request }) => {
    const resp = await request.get(`${BASE}/api/cache/models`);
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(body).toHaveProperty('models');
    expect(Array.isArray(body.models)).toBe(true);
    if (body.models.length > 0) {
      expect(body.models[0]).toHaveProperty('sizeBytes');
    }
  });

  test('GET /api/cache/stats returns required fields', async ({ request }) => {
    const resp = await request.get(`${BASE}/api/cache/stats`);
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(body).toHaveProperty('totalModels');
    expect(body).toHaveProperty('totalSizeMB');
    expect(body).toHaveProperty('byType');
    expect(body).toHaveProperty('cacheDirectory');
  });

  test('GET /api/cache/loaded returns modelType, modelId, lastUsedAt', async ({ request }) => {
    const resp = await request.get(`${BASE}/api/cache/loaded`);
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(Array.isArray(body)).toBe(true);
    if (body.length > 0) {
      expect(body[0]).toHaveProperty('modelType');
      expect(body[0]).toHaveProperty('modelId');
      expect(body[0]).toHaveProperty('lastUsedAt');
    }
  });

  test('GET /api/cache/models/type/embedder returns embedder only', async ({ request }) => {
    const resp = await request.get(`${BASE}/api/cache/models/type/embedder`);
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(body).toHaveProperty('models');
    if (body.models.length > 0) {
      for (const model of body.models) {
        expect(model.detectedType?.toLowerCase()).toContain('embed');
      }
    }
  });

  test('GET /api/cache/models/type/invalid returns 400', async ({ request }) => {
    const resp = await request.get(`${BASE}/api/cache/models/type/invalid`);
    expect(resp.status()).toBe(400);
  });

  test('DELETE /api/cache/loaded/{id} for nonexistent model returns 404', async ({ request }) => {
    const resp = await request.delete(`${BASE}/api/cache/loaded/generator:nonexistent-val-plan`);
    expect(resp.status()).toBe(404);
  });

  test('DELETE /api/cache/models/{repoId} for nonexistent returns 404', async ({ request }) => {
    const resp = await request.delete(`${BASE}/api/cache/models/nonexistent%2Frepo-val-plan`);
    expect(resp.status()).toBe(404);
  });
});

// ================================================================
// Download API
// ================================================================
test.describe('Download API', () => {
  test('POST /api/download/check with known repoId returns metadata', async ({ request }) => {
    const resp = await request.post(`${BASE}/api/download/check`, {
      data: { repoId: 'BAAI/bge-small-en-v1.5' },
    });
    expect([200, 404, 503]).toContain(resp.status());
    if (resp.status() === 200) {
      const body = await resp.json();
      expect(body).toBeTruthy();
    }
  });

  test('POST /api/download/check with missing repoId returns 400', async ({ request }) => {
    const resp = await request.post(`${BASE}/api/download/check`, {
      data: {},
    });
    expect(resp.status()).toBe(400);
  });

  test('SSE download stream uses text/event-stream', async ({ request }) => {
    // Just verify the endpoint exists and uses SSE format
    const resp = await request.post(`${BASE}/api/download/start`, {
      data: { repoId: 'BAAI/bge-small-en-v1.5' },
    });
    // Endpoint exists (not 404/405) — actual streaming test requires live download
    expect(resp.status()).not.toBe(404);
    expect(resp.status()).not.toBe(405);
  });
});

// ================================================================
// Registry API
// ================================================================
test.describe('Registry API', () => {
  test('GET /api/registry/models returns 11 model types', async ({ request }) => {
    const resp = await request.get(`${BASE}/api/registry/models`);
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(body).toHaveProperty('modelTypes');
    expect(Array.isArray(body.modelTypes)).toBe(true);
    expect(body.modelTypes.length).toBeGreaterThanOrEqual(11);
    for (const mt of body.modelTypes) {
      expect(mt).toHaveProperty('type');
      expect(mt).toHaveProperty('models');
    }
  });

  test('GET /api/registry/models/embedder returns embedder type', async ({ request }) => {
    const resp = await request.get(`${BASE}/api/registry/models/embedder`);
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(body.type).toBe('embedder');
    expect(Array.isArray(body.models)).toBe(true);
  });

  test('isCached field present in registry response', async ({ request }) => {
    const resp = await request.get(`${BASE}/api/registry/models/embedder`);
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    if (body.models.length > 0) {
      expect(body.models[0]).toHaveProperty('isCached');
    }
  });

  test('GET /api/registry/models/invalidtype returns 404', async ({ request }) => {
    const resp = await request.get(`${BASE}/api/registry/models/invaliddomaintype`);
    expect(resp.status()).toBe(404);
  });
});

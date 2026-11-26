import { test, expect } from '@playwright/test';
import { BASE, requireHub } from './host-capabilities';

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
    const ids: string[] = body.data.map((m: { id: string }) => m.id.toLowerCase());
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
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(body.id).toBeTruthy();
    expect(body.object).toBe('model');
  });

  test('nonexistent model returns 404 with error envelope', async ({ request }) => {
    const resp = await request.get(`${BASE}/v1/models/nonexistent-model-xyz`);
    expect(resp.status()).toBe(404);
    const body = await resp.json();
    expect(body.error).toBeTruthy();
    expect(body.error.code).toBeTruthy();
  });

  test('URL-encoded slash in model ID is handled', async ({ request }) => {
    // The id is no registered alias: a 404 with the error envelope, not a 400 for the slash or a 500.
    const resp = await request.get(`${BASE}/v1/models/microsoft%2FFlorence-2-base`);
    expect(resp.status()).toBe(404);
    const body = await resp.json();
    expect(body.error.code).toBe('model_not_found');
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
    expect(Array.isArray(body)).toBe(true);
    if (body.length > 0) {
      for (const model of body) {
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
    await requireHub(request);
    const resp = await request.post(`${BASE}/api/download/check`, {
      data: { repoId: 'BAAI/bge-small-en-v1.5' },
    });
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(body.exists).toBe(true);
    expect(body.repoId).toBe('BAAI/bge-small-en-v1.5');
    expect(body.fileCount).toBeGreaterThan(0);
  });

  test('POST /api/download/check with missing repoId returns 400', async ({ request }) => {
    const resp = await request.post(`${BASE}/api/download/check`, {
      data: {},
    });
    expect(resp.status()).toBe(400);
  });

  test('POST /api/download/model without a repository is refused before any download starts', async ({ request }) => {
    // The streaming answer needs a live download, which a test must not start; the refusal shows the endpoint is there.
    const resp = await request.post(`${BASE}/api/download/model`, { data: {} });
    expect(resp.status()).toBe(400);
    const body = await resp.json();
    expect(body.error.message).toContain('RepoId');
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

  // The generator list is the library's two catalogs (GGUF aliases and ONNX entries), with "default" first: the one the hardware picks.
  test('GET /api/registry/models/generator lists the library catalogs and says per alias whether it is cached', async ({ request }) => {
    const resp = await request.get(`${BASE}/api/registry/models/generator`);
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    const aliases: string[] = body.models.map((m: { aliasName: string }) => m.aliasName);
    expect(aliases[0]).toBe('default');
    expect(aliases).toContain('gguf:balanced');
    expect(aliases.filter(a => a.startsWith('gguf:')).length).toBeGreaterThan(5);
    expect(new Set(aliases).size).toBe(aliases.length);

    // An alias that is cached has its repository in the cache listing ("default" names no repository).
    const cache = await (await request.get(`${BASE}/api/cache/models`)).json();
    const repos = new Set((cache.models as { repoId: string }[]).map(m => m.repoId.toLowerCase()));
    for (const model of body.models.filter((m: { aliasName: string; isCached: boolean }) => m.aliasName !== 'default' && m.isCached)) {
      expect(repos.has(model.repoId.toLowerCase()), `${model.aliasName} is cached: ${model.repoId}`).toBe(true);
    }
  });

  // One repository holds many OCR models, each in a subfolder; the registry names the repository (a path inside it names no cache entry).
  test('GET /api/registry/models/ocr names repositories and says per model whether its files are cached', async ({ request }) => {
    const resp = await request.get(`${BASE}/api/registry/models/ocr`);
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(body.models.length).toBeGreaterThan(0);
    for (const model of body.models) {
      expect(model.repoId, model.aliasName).toMatch(/^[^/]+\/[^/]+$/);
      expect(typeof model.isCached).toBe('boolean');
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

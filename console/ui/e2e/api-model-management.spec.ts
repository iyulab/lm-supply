import { test, expect } from './fixtures/base.fixture';

// ================================================================
// API Model Management Endpoints
// ================================================================

test.describe('API Model Management', () => {
  // GET /api/registry/models returns all domain types
  test('GET /api/registry/models returns 11 domain types', async ({ request }) => {
    const response = await request.get('/api/registry/models');
    expect(response.status()).toBe(200);

    const body = await response.json();
    expect(body).toHaveProperty('modelTypes');
    expect(Array.isArray(body.modelTypes)).toBe(true);

    // Should have 11 AI domain types
    const types = body.modelTypes.map((t: { type: string }) => t.type);
    expect(types.length).toBeGreaterThanOrEqual(11);

    // Spot-check key domains
    expect(types).toContain('generator');
    expect(types).toContain('embedder');
    expect(types).toContain('transcriber');

    // Each type should have models array
    for (const modelType of body.modelTypes) {
      expect(modelType).toHaveProperty('type');
      expect(modelType).toHaveProperty('displayName');
      expect(modelType).toHaveProperty('models');
      expect(Array.isArray(modelType.models)).toBe(true);
    }
  });

  // GET /api/registry/models/embedder returns only embedder models
  test('GET /api/registry/models/embedder returns embedder models only', async ({ request }) => {
    const response = await request.get('/api/registry/models/embedder');
    expect(response.status()).toBe(200);

    const body = await response.json();
    expect(body.type).toBe('embedder');
    expect(body).toHaveProperty('displayName');
    expect(Array.isArray(body.models)).toBe(true);
    expect(body.models.length).toBeGreaterThan(0);

    // Each model should have alias and repoId
    for (const model of body.models) {
      expect(model).toHaveProperty('aliasName');
      expect(model).toHaveProperty('repoId');
      expect(model).toHaveProperty('isCached');
    }
  });

  // GET /api/cache/models returns cached model list
  test('GET /api/cache/models returns cached models array', async ({ request }) => {
    const response = await request.get('/api/cache/models');
    expect(response.status()).toBe(200);

    const body = await response.json();
    expect(body).toHaveProperty('models');
    expect(Array.isArray(body.models)).toBe(true);

    // If cached models exist, verify structure
    if (body.models.length > 0) {
      const model = body.models[0];
      expect(model).toHaveProperty('repoId');
      expect(model).toHaveProperty('sizeBytes');
      expect(model).toHaveProperty('fileCount');
      expect(model).toHaveProperty('detectedType');
    }
  });

  // GET /api/cache/loaded returns loaded models
  test('GET /api/cache/loaded returns loaded models array', async ({ request }) => {
    const response = await request.get('/api/cache/loaded');
    expect(response.status()).toBe(200);

    const body = await response.json();
    expect(Array.isArray(body)).toBe(true);

    // If models are loaded, verify structure
    if (body.length > 0) {
      const model = body[0];
      expect(model).toHaveProperty('modelType');
      expect(model).toHaveProperty('modelId');
    }
  });

  // GET /api/cache/stats returns cache statistics
  test('GET /api/cache/stats returns cache statistics', async ({ request }) => {
    const response = await request.get('/api/cache/stats');
    expect(response.status()).toBe(200);

    const body = await response.json();
    expect(body).toHaveProperty('totalModels');
    expect(body).toHaveProperty('totalSizeMB');
    expect(body).toHaveProperty('cacheDirectory');
    expect(body).toHaveProperty('byType');

    expect(typeof body.totalModels).toBe('number');
    expect(typeof body.totalSizeMB).toBe('number');
    expect(typeof body.cacheDirectory).toBe('string');
    expect(body.totalModels).toBeGreaterThanOrEqual(0);
  });

  // POST /api/download/check with valid repoId
  test('POST /api/download/check returns model info for valid repo', async ({ request }) => {
    const response = await request.post('/api/download/check', {
      data: { repoId: 'sentence-transformers/all-MiniLM-L6-v2' },
    });
    expect(response.status()).toBe(200);

    const body = await response.json();
    expect(body).toHaveProperty('exists');
    expect(body.exists).toBe(true);
    expect(body).toHaveProperty('detectedType');
    expect(body).toHaveProperty('fileCount');
    expect(body).toHaveProperty('totalSizeBytes');
    expect(body.fileCount).toBeGreaterThan(0);
  });

  // POST /api/download/check with invalid repoId
  test('POST /api/download/check returns error for nonexistent repo', async ({ request }) => {
    const response = await request.post('/api/download/check', {
      data: { repoId: 'nonexistent/fake-model-999' },
    });

    // Should return error status or exists=false
    const body = await response.json();
    if (response.status() === 200) {
      expect(body.exists).toBe(false);
    } else {
      expect(response.status()).toBeGreaterThanOrEqual(400);
    }
  });

  // GET /v1/models returns OpenAI-compatible model list
  test('GET /v1/models returns OpenAI-compatible model list', async ({ request }) => {
    const response = await request.get('/v1/models');
    expect(response.status()).toBe(200);

    const body = await response.json();
    expect(body).toHaveProperty('object', 'list');
    expect(body).toHaveProperty('data');
    expect(Array.isArray(body.data)).toBe(true);

    // Should have at least some well-known aliases
    expect(body.data.length).toBeGreaterThan(0);

    // Each model should follow OpenAI format
    const model = body.data[0];
    expect(model).toHaveProperty('id');
    expect(model).toHaveProperty('object', 'model');
    expect(model).toHaveProperty('created');
    expect(model).toHaveProperty('ownedBy');
  });
});

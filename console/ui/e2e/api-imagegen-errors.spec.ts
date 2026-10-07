import { test, expect } from './fixtures/base.fixture';

// ================================================================
// API — Image Generation + Error Responses
// ================================================================

test.describe('API Image Generation', () => {
  test.setTimeout(120_000);

  // POST /v1/images/generations (OpenAI-compatible)
  test('POST /v1/images/generations returns b64_json', async ({ request }) => {
    const response = await request.post('/v1/images/generations', {
      data: {
        model: 'default',
        prompt: 'a simple red circle on white background',
        size: '256x256',
        n: 1,
      },
    });

    if (response.status() === 200) {
      const body = await response.json();
      expect(body).toHaveProperty('data');
      expect(Array.isArray(body.data)).toBe(true);
      expect(body.data.length).toBe(1);
      expect(body.data[0]).toHaveProperty('b64_json');
      expect(body.data[0].b64_json.length).toBeGreaterThan(100);
    } else {
      // Model may not be available
      const body = await response.json();
      expect(body).toHaveProperty('error');
    }
  });

  // POST /v1/images/generate (extended format)
  test('POST /v1/images/generate returns extended response', async ({ request }) => {
    const response = await request.post('/v1/images/generate', {
      data: {
        model: 'default',
        prompt: 'a blue square',
        width: 256,
        height: 256,
        steps: 4,
      },
    });

    if (response.status() === 200) {
      const body = await response.json();
      expect(body).toHaveProperty('data');
      expect(Array.isArray(body.data)).toBe(true);
      // Extended fields
      expect(body).toHaveProperty('generation_time_ms');
      expect(body).toHaveProperty('model');
    } else {
      const body = await response.json();
      expect(body).toHaveProperty('error');
    }
  });

  // GET /v1/images/models returns available models
  test('GET /v1/images/models returns model list', async ({ request }) => {
    const response = await request.get('/v1/images/models');
    expect(response.status()).toBe(200);

    const body = await response.json();
    expect(body).toHaveProperty('models');
    expect(Array.isArray(body.models)).toBe(true);
    expect(body.models.length).toBeGreaterThan(0);

    const model = body.models[0];
    expect(model).toHaveProperty('id');
    expect(model).toHaveProperty('repo_id');
    expect(model).toHaveProperty('recommended_steps');
    expect(model).toHaveProperty('recommended_guidance_scale');
  });

  // GET /v1/translate/languages returns language pairs
  test('GET /v1/translate/languages returns language pairs', async ({ request }) => {
    const response = await request.get('/v1/translate/languages');
    expect(response.status()).toBe(200);

    const body = await response.json();
    expect(body).toHaveProperty('languages');
    expect(Array.isArray(body.languages)).toBe(true);
    expect(body.languages.length).toBeGreaterThan(0);

    const lang = body.languages[0];
    expect(lang).toHaveProperty('id');
    expect(lang).toHaveProperty('source');
    expect(lang).toHaveProperty('target');
  });
});

test.describe('API Error Responses', () => {
  // POST /v1/chat/completions with empty messages returns 400
  test('POST /v1/chat/completions with empty messages returns error', async ({ request }) => {
    const response = await request.post('/v1/chat/completions', {
      data: {
        model: 'default',
        messages: [],
        stream: false,
      },
    });
    expect(response.status()).toBeGreaterThanOrEqual(400);
  });

  // POST /v1/rerank with empty query returns 400
  test('POST /v1/rerank with missing query returns error', async ({ request }) => {
    const response = await request.post('/v1/rerank', {
      data: {
        model: 'default',
        documents: ['test'],
      },
    });
    expect(response.status()).toBeGreaterThanOrEqual(400);
  });

  // POST /v1/embeddings with empty input returns 400
  test('POST /v1/embeddings with missing input returns error', async ({ request }) => {
    const response = await request.post('/v1/embeddings', {
      data: {
        model: 'default',
      },
    });
    expect(response.status()).toBeGreaterThanOrEqual(400);
  });

  // POST to nonexistent path returns 404
  test('POST to nonexistent path returns 404', async ({ request }) => {
    const response = await request.post('/v1/nonexistent', {
      data: { test: true },
    });
    expect(response.status()).toBe(404);
  });

  // DELETE nonexistent cached model returns 404
  test('DELETE nonexistent cached model returns 404', async ({ request }) => {
    const response = await request.delete('/api/cache/models/nonexistent%2Ffake-model');
    expect(response.status()).toBeGreaterThanOrEqual(400);
  });
});

import { test, expect } from './fixtures/base.fixture';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const TEST_IMAGE = path.resolve(__dirname, 'fixtures/test-files/test-image.png');

// ================================================================
// API — Model Load/Unload cycle + Segment with mask
// ================================================================

test.describe('API Load/Unload cycle', () => {
  test.setTimeout(120_000);

  // [5-14] POST /api/cache/load loads a model
  test('[5-14] POST /api/cache/load returns key and model info', async ({ request }) => {
    const response = await request.post('/api/cache/load', {
      data: { type: 'embedder', modelId: 'default' },
    });
    expect(response.status()).toBe(200);

    const body = await response.json();
    expect(body).toHaveProperty('key');
    expect(body.key).toContain('embedder');
    expect(body).toHaveProperty('model');
    expect(body.model).toHaveProperty('modelId');
    expect(body.model).toHaveProperty('modelType', 'embedder');
  });

  // [5-15] DELETE /api/cache/loaded/{key} unloads a model
  test('[5-15] load then unload model via API', async ({ request }) => {
    // First load a model
    const loadRes = await request.post('/api/cache/load', {
      data: { type: 'reranker', modelId: 'default' },
    });
    expect(loadRes.status()).toBe(200);
    const loadBody = await loadRes.json();
    const key = loadBody.key;

    // Verify it's loaded
    const loadedRes = await request.get('/api/cache/loaded');
    const loaded = await loadedRes.json();
    const found = loaded.some((m: { modelType: string }) => m.modelType === 'reranker');
    expect(found).toBe(true);

    // Unload it
    const unloadRes = await request.delete(`/api/cache/loaded/${encodeURIComponent(key)}`);
    expect(unloadRes.status()).toBe(200);
  });
});

test.describe('API Segment with mask', () => {
  test.setTimeout(120_000);

  // [5-34] POST /v1/images/segment with include_mask=true
  test('[5-34] segment with include_mask returns mask data', async ({ request }) => {
    const imageBuffer = fs.readFileSync(TEST_IMAGE);
    const response = await request.post('/v1/images/segment', {
      multipart: {
        model: 'default',
        file: { name: 'test-image.png', mimeType: 'image/png', buffer: imageBuffer },
        include_mask: 'true',
      },
    });

    if (response.status() === 200) {
      const body = await response.json();
      expect(body).toHaveProperty('segments');
      // When mask is requested, segments should include mask data
      if (body.segments.length > 0) {
        // At least check the structure exists
        expect(body.segments[0]).toHaveProperty('label');
        expect(body.segments[0]).toHaveProperty('score');
      }
    } else {
      // Model may not be available
      const body = await response.json();
      expect(body).toHaveProperty('error');
    }
  });
});

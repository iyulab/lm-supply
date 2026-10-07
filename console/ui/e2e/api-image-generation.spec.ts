import { test, expect } from '@playwright/test';

const BASE = process.env.API_BASE_URL || 'http://localhost:5000';

// 1x1 red pixel PNG as buffer for multipart image uploads
const PNG_BUFFER = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg==',
  'base64'
);

// ================================================================
// POST /v1/images/generations
// ================================================================
test.describe('Image Generations (OpenAI compat)', () => {
  test('response format: created, data[0].b64_json or url', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/generations`, {
      data: { prompt: 'A cute cat', model: 'default', size: '256x256' },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      expect(typeof body.created).toBe('number');
      expect(Array.isArray(body.data)).toBe(true);
      expect(body.data[0].b64_json || body.data[0].url).toBeTruthy();
    } else {
      expect([404, 503]).toContain(resp.status());
    }
  });

  test('response_format url returns data[0].url with localhost path', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/generations`, {
      data: { prompt: 'A cat', model: 'default', size: '256x256', response_format: 'url' },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      expect(body.data[0].url).toBeTruthy();
      expect(body.data[0].url).toContain('localhost');
    }
  });

  test('URL from response_format url is accessible', async ({ request }) => {
    const gen = await request.post(`${BASE}/v1/images/generations`, {
      data: { prompt: 'A dot', model: 'default', size: '256x256', response_format: 'url' },
    });
    if (gen.status() === 200) {
      const body = await gen.json();
      const url = body.data[0].url;
      if (url) {
        const imgResp = await request.get(url);
        expect(imgResp.status()).toBe(200);
        expect(imgResp.headers()['content-type']).toContain('image');
      }
    }
  });

  test('n=3 returns 3 data items', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/generations`, {
      data: { prompt: 'A dot', model: 'default', size: '256x256', n: 3 },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      expect(body.data.length).toBe(3);
    }
  });

  test('n > 4 is clamped to 4', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/generations`, {
      data: { prompt: 'A dot', model: 'default', size: '256x256', n: 10 },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      expect(body.data.length).toBeLessThanOrEqual(4);
    } else {
      // 400 is also acceptable (reject n > 4)
      expect([200, 400, 404, 503]).toContain(resp.status());
    }
  });

  test('size with non-multiple of 8 returns 400', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/generations`, {
      data: { prompt: 'test', model: 'default', size: '255x255' },
    });
    expect(resp.status()).toBe(400);
  });

  test('size > 2048 returns 400', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/generations`, {
      data: { prompt: 'test', model: 'default', size: '2304x2304' },
    });
    expect(resp.status()).toBe(400);
  });

  test('missing prompt returns 400', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/generations`, {
      data: { model: 'default', size: '256x256' },
    });
    expect(resp.status()).toBe(400);
  });

  test('steps parameter accepted (not 422)', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/generations`, {
      data: { prompt: 'test', model: 'default', size: '256x256', steps: 4 },
    });
    expect(resp.status()).not.toBe(422);
  });

  test('seed parameter accepted (not 422)', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/generations`, {
      data: { prompt: 'test', model: 'default', size: '256x256', seed: 42 },
    });
    expect(resp.status()).not.toBe(422);
  });
});

// ================================================================
// POST /v1/images/edits
// ================================================================
test.describe('Image Edits', () => {
  test('edits endpoint returns OpenAI-compatible response (not 404)', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/edits`, {
      multipart: {
        image: { name: 'test.png', mimeType: 'image/png', buffer: PNG_BUFFER },
        prompt: 'Add a hat',
        model: 'default',
      },
    });
    expect(resp.status()).not.toBe(404);
    expect(resp.status()).not.toBe(405);
  });

  test('edits without prompt returns 400', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/edits`, {
      multipart: {
        image: { name: 'test.png', mimeType: 'image/png', buffer: PNG_BUFFER },
      },
    });
    expect(resp.status()).toBe(400);
  });

  test('edits with response_format url returns url (not 422)', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/edits`, {
      multipart: {
        image: { name: 'test.png', mimeType: 'image/png', buffer: PNG_BUFFER },
        prompt: 'Test edit',
        response_format: 'url',
      },
    });
    expect(resp.status()).not.toBe(422);
  });
});

// ================================================================
// POST /v1/images/variations
// ================================================================
test.describe('Image Variations', () => {
  test('variations endpoint returns OpenAI-compatible response (not 404)', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/variations`, {
      multipart: {
        image: { name: 'test.png', mimeType: 'image/png', buffer: PNG_BUFFER },
        model: 'default',
      },
    });
    expect(resp.status()).not.toBe(404);
    expect(resp.status()).not.toBe(405);
  });

  test('variations without explicit prompt still works (not 400)', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/variations`, {
      multipart: {
        image: { name: 'test.png', mimeType: 'image/png', buffer: PNG_BUFFER },
      },
    });
    // No prompt required — should use default
    expect(resp.status()).not.toBe(400);
  });
});

// ================================================================
// TempFileService
// ================================================================
test.describe('TempFileService', () => {
  test('generated URL returns 200 with image content-type', async ({ request }) => {
    const gen = await request.post(`${BASE}/v1/images/generations`, {
      data: { prompt: 'dot', model: 'default', size: '256x256', response_format: 'url' },
    });
    if (gen.status() === 200) {
      const body = await gen.json();
      const url = body.data[0].url;
      if (url) {
        const fileResp = await request.get(url);
        expect(fileResp.status()).toBe(200);
        expect(fileResp.headers()['content-type']).toContain('image');
      }
    }
  });

  test('nonexistent temp file ID returns 404', async ({ request }) => {
    const resp = await request.get(`${BASE}/v1/images/files/nonexistent-id-e2e`);
    expect(resp.status()).toBe(404);
  });
});

// ================================================================
// POST /v1/images/generate (Extended API)
// ================================================================
test.describe('Extended Image Generation', () => {
  test('/v1/images/generate response includes extended fields', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/generate`, {
      data: { prompt: 'Sunset', model: 'default' },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      expect(body).toHaveProperty('id');
      expect(body).toHaveProperty('model');
      expect(body).toHaveProperty('created');
      expect(body).toHaveProperty('generation_time_ms');
      expect(Array.isArray(body.data)).toBe(true);
    } else {
      expect([404, 503]).toContain(resp.status());
    }
  });

  test('generation_time_ms is non-negative integer', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/generate`, {
      data: { prompt: 'test', model: 'default' },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      expect(typeof body.generation_time_ms).toBe('number');
      expect(body.generation_time_ms).toBeGreaterThanOrEqual(0);
    }
  });

  test('seed value is returned in response', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/generate`, {
      data: { prompt: 'test', model: 'default', seed: 42 },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      if (body.data && body.data.length > 0) {
        expect(body.data[0]).toHaveProperty('seed');
      }
    }
  });
});

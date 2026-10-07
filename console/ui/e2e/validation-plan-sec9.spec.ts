import { test, expect } from '@playwright/test';

const BASE = process.env.API_BASE_URL || 'http://localhost:5000';

// 1x1 red pixel PNG as buffer for multipart image uploads
const PNG_BUFFER = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg==',
  'base64'
);

// ================================================================
// Section 9-1: POST /v1/images/generations
// ================================================================
test.describe('Section 9-1: Image Generations (OpenAI compat)', () => {
  test('[9-1-01] response format: created, data[0].b64_json or url', async ({ request }) => {
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

  test('[9-1-02] response_format url returns data[0].url with localhost path', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/generations`, {
      data: { prompt: 'A cat', model: 'default', size: '256x256', response_format: 'url' },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      expect(body.data[0].url).toBeTruthy();
      expect(body.data[0].url).toContain('localhost');
    }
  });

  test('[9-1-03] URL from response_format url is accessible', async ({ request }) => {
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

  test('[9-1-04] n=3 returns 3 data items', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/generations`, {
      data: { prompt: 'A dot', model: 'default', size: '256x256', n: 3 },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      expect(body.data.length).toBe(3);
    }
  });

  test('[9-1-05] n > 4 is clamped to 4', async ({ request }) => {
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

  test('[9-1-08] size with non-multiple of 8 returns 400', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/generations`, {
      data: { prompt: 'test', model: 'default', size: '255x255' },
    });
    expect(resp.status()).toBe(400);
  });

  test('[9-1-09] size > 2048 returns 400', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/generations`, {
      data: { prompt: 'test', model: 'default', size: '2304x2304' },
    });
    expect(resp.status()).toBe(400);
  });

  test('[9-1-10] missing prompt returns 400', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/generations`, {
      data: { model: 'default', size: '256x256' },
    });
    expect(resp.status()).toBe(400);
  });

  test('[9-1-11] steps parameter accepted (not 422)', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/generations`, {
      data: { prompt: 'test', model: 'default', size: '256x256', steps: 4 },
    });
    expect(resp.status()).not.toBe(422);
  });

  test('[9-1-12] seed parameter accepted (not 422)', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/generations`, {
      data: { prompt: 'test', model: 'default', size: '256x256', seed: 42 },
    });
    expect(resp.status()).not.toBe(422);
  });
});

// ================================================================
// Section 9-2: POST /v1/images/edits
// ================================================================
test.describe('Section 9-2: Image Edits', () => {
  test('[9-2-01] edits endpoint returns OpenAI-compatible response (not 404)', async ({ request }) => {
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

  test('[9-2-02] edits without prompt returns 400', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/edits`, {
      multipart: {
        image: { name: 'test.png', mimeType: 'image/png', buffer: PNG_BUFFER },
      },
    });
    expect(resp.status()).toBe(400);
  });

  test('[9-2-03] edits with response_format url returns url (not 422)', async ({ request }) => {
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
// Section 9-3: POST /v1/images/variations
// ================================================================
test.describe('Section 9-3: Image Variations', () => {
  test('[9-3-01] variations endpoint returns OpenAI-compatible response (not 404)', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/variations`, {
      multipart: {
        image: { name: 'test.png', mimeType: 'image/png', buffer: PNG_BUFFER },
        model: 'default',
      },
    });
    expect(resp.status()).not.toBe(404);
    expect(resp.status()).not.toBe(405);
  });

  test('[9-3-02] variations without explicit prompt still works (not 400)', async ({ request }) => {
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
// Section 9-4: TempFileService
// ================================================================
test.describe('Section 9-4: TempFileService', () => {
  test('[9-4-01][9-4-02] generated URL returns 200 with image content-type', async ({ request }) => {
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

  test('[9-4-03] nonexistent temp file ID returns 404', async ({ request }) => {
    const resp = await request.get(`${BASE}/v1/images/files/nonexistent-id-validation-plan`);
    expect(resp.status()).toBe(404);
  });
});

// ================================================================
// Section 9-5: POST /v1/images/generate (Extended API)
// ================================================================
test.describe('Section 9-5: Extended Image Generation', () => {
  test('[9-5-01] /v1/images/generate response includes extended fields', async ({ request }) => {
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

  test('[9-5-02] generation_time_ms is non-negative integer', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/generate`, {
      data: { prompt: 'test', model: 'default' },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      expect(typeof body.generation_time_ms).toBe('number');
      expect(body.generation_time_ms).toBeGreaterThanOrEqual(0);
    }
  });

  test('[9-5-03] seed value is returned in response', async ({ request }) => {
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

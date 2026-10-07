import { test, expect } from '@playwright/test';

const BASE = process.env.API_BASE_URL || 'http://localhost:5000';

// ================================================================
// Section 4-1: POST /v1/embeddings
// ================================================================
test.describe('Section 4-1: Embeddings', () => {
  test('[4-1-01] response structure: object list, data[0].embedding, usage', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/embeddings`, {
      data: { model: 'default', input: 'Hello world' },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      expect(body.object).toBe('list');
      expect(body.data[0].object).toBe('embedding');
      expect(Array.isArray(body.data[0].embedding)).toBe(true);
      expect(body.data[0].embedding.length).toBeGreaterThan(0);
      expect(body).toHaveProperty('usage');
    } else {
      expect([404, 503]).toContain(resp.status());
    }
  });

  test('[4-1-02] array input returns data array of same length', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/embeddings`, {
      data: { model: 'default', input: ['text1', 'text2'] },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      expect(body.data.length).toBe(2);
      expect(body.data[0].index).toBe(0);
      expect(body.data[1].index).toBe(1);
    } else {
      expect([404, 503]).toContain(resp.status());
    }
  });

  test('[4-1-03] empty input returns 400', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/embeddings`, {
      data: { model: 'default', input: '' },
    });
    expect(resp.status()).toBe(400);
  });

  test('[4-1-04] encoding_format float returns float array', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/embeddings`, {
      data: { model: 'default', input: 'hello', encoding_format: 'float' },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      expect(Array.isArray(body.data[0].embedding)).toBe(true);
      expect(typeof body.data[0].embedding[0]).toBe('number');
    } else {
      expect([404, 503]).toContain(resp.status());
    }
  });

  test('[4-1-05] encoding_format base64 returns string', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/embeddings`, {
      data: { model: 'default', input: 'hello', encoding_format: 'base64' },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      expect(typeof body.data[0].embedding).toBe('string');
    } else {
      expect([404, 503]).toContain(resp.status());
    }
  });

  test('[4-1-06] dimensions parameter accepted (not 422)', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/embeddings`, {
      data: { model: 'default', input: 'hello', dimensions: 64 },
    });
    expect(resp.status()).not.toBe(422);
    if (resp.status() === 200) {
      const body = await resp.json();
      expect(body.data[0].embedding.length).toBeLessThanOrEqual(64);
    }
  });

  test('[4-1-07] dimensions > actual returns full vector without error', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/embeddings`, {
      data: { model: 'default', input: 'hello', dimensions: 99999 },
    });
    expect(resp.status()).not.toBe(422);
    expect(resp.status()).not.toBe(400);
  });

  test('[4-1-08] usage.total_tokens is non-negative integer', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/embeddings`, {
      data: { model: 'default', input: 'hello' },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      expect(typeof body.usage.total_tokens).toBe('number');
      expect(body.usage.total_tokens).toBeGreaterThanOrEqual(0);
    }
  });
});

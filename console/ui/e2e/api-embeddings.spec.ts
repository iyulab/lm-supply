import { test, expect } from '@playwright/test';
import { BASE, requireCachedModel } from './host-capabilities';

// ================================================================
// POST /v1/embeddings
// ================================================================
test.describe('Embeddings', () => {
  test('response structure: object list, data[0].embedding, usage', async ({ request }) => {
    await requireCachedModel(request, 'embedder');
    const resp = await request.post(`${BASE}/v1/embeddings`, {
      data: { model: 'default', input: 'Hello world' },
    });
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(body.object).toBe('list');
    expect(body.data[0].object).toBe('embedding');
    expect(Array.isArray(body.data[0].embedding)).toBe(true);
    expect(body.data[0].embedding.length).toBeGreaterThan(0);
    expect(body).toHaveProperty('usage');
  });

  test('array input returns data array of same length', async ({ request }) => {
    await requireCachedModel(request, 'embedder');
    const resp = await request.post(`${BASE}/v1/embeddings`, {
      data: { model: 'default', input: ['text1', 'text2'] },
    });
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(body.data.length).toBe(2);
    expect(body.data[0].index).toBe(0);
    expect(body.data[1].index).toBe(1);
  });

  test('empty input returns 400', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/embeddings`, {
      data: { model: 'default', input: '' },
    });
    expect(resp.status()).toBe(400);
  });

  test('encoding_format float returns float array', async ({ request }) => {
    await requireCachedModel(request, 'embedder');
    const resp = await request.post(`${BASE}/v1/embeddings`, {
      data: { model: 'default', input: 'hello', encoding_format: 'float' },
    });
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(Array.isArray(body.data[0].embedding)).toBe(true);
    expect(typeof body.data[0].embedding[0]).toBe('number');
  });

  test('encoding_format base64 returns string', async ({ request }) => {
    await requireCachedModel(request, 'embedder');
    const resp = await request.post(`${BASE}/v1/embeddings`, {
      data: { model: 'default', input: 'hello', encoding_format: 'base64' },
    });
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(typeof body.data[0].embedding).toBe('string');
  });

  test('dimensions parameter truncates the vector', async ({ request }) => {
    await requireCachedModel(request, 'embedder');
    const resp = await request.post(`${BASE}/v1/embeddings`, {
      data: { model: 'default', input: 'hello', dimensions: 64 },
    });
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(body.data[0].embedding.length).toBe(64);
  });

  test('dimensions > actual returns full vector without error', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/embeddings`, {
      data: { model: 'default', input: 'hello', dimensions: 99999 },
    });
    expect(resp.status()).not.toBe(422);
    expect(resp.status()).not.toBe(400);
  });

  test('usage.total_tokens is non-negative integer', async ({ request }) => {
    await requireCachedModel(request, 'embedder');
    const resp = await request.post(`${BASE}/v1/embeddings`, {
      data: { model: 'default', input: 'hello' },
    });
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(Number.isInteger(body.usage.total_tokens)).toBe(true);
    expect(body.usage.total_tokens).toBeGreaterThanOrEqual(0);
  });
});

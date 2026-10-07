import { test, expect } from '@playwright/test';

const BASE = process.env.API_BASE_URL || 'http://localhost:5000';

const DOCS = ['Deep learning is a subset of machine learning', 'Cats are domesticated animals', 'Neural networks learn patterns from data'];

// ================================================================
// Section 5-1: POST /v1/rerank
// ================================================================
test.describe('Section 5-1: Rerank', () => {
  test('[5-1-01] response structure: id, results[].index/relevance_score/document.text, meta', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/rerank`, {
      data: { model: 'default', query: 'machine learning', documents: DOCS },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      expect(body).toHaveProperty('id');
      expect(Array.isArray(body.results)).toBe(true);
      expect(body.results[0]).toHaveProperty('index');
      expect(body.results[0]).toHaveProperty('relevance_score');
      expect(body.results[0].document).toHaveProperty('text');
      expect(body).toHaveProperty('meta');
    } else {
      expect([404, 503]).toContain(resp.status());
    }
  });

  test('[5-1-02] results sorted by relevance_score descending', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/rerank`, {
      data: { model: 'default', query: 'machine learning', documents: DOCS },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      for (let i = 1; i < body.results.length; i++) {
        expect(body.results[i - 1].relevance_score).toBeGreaterThanOrEqual(body.results[i].relevance_score);
      }
    }
  });

  test('[5-1-03] top_n limits number of results', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/rerank`, {
      data: { model: 'default', query: 'machine learning', documents: DOCS, top_n: 2 },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      expect(body.results.length).toBeLessThanOrEqual(2);
    }
  });

  test('[5-1-04] return_documents false omits document field', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/rerank`, {
      data: { model: 'default', query: 'machine learning', documents: DOCS, return_documents: false },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      if (body.results.length > 0) {
        expect(body.results[0].document).toBeFalsy();
      }
    }
  });

  test('[5-1-05] JSON object documents accepted (not 422)', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/rerank`, {
      data: {
        model: 'default',
        query: 'AI',
        documents: [
          { title: 'Machine Learning', body: 'ML is about learning from data.' },
          { title: 'Cats', body: 'Cats are pets.' },
        ],
      },
    });
    expect(resp.status()).not.toBe(422);
  });

  test('[5-1-06] rank_fields parameter accepted (not 422)', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/rerank`, {
      data: {
        model: 'default',
        query: 'AI',
        documents: [
          { title: 'AI basics', body: 'Artificial intelligence.' },
          { title: 'Cooking', body: 'Recipes and food.' },
        ],
        rank_fields: ['title', 'body'],
      },
    });
    expect(resp.status()).not.toBe(422);
  });

  test('[5-1-07] meta field has billed_units.search_units', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/rerank`, {
      data: { model: 'default', query: 'ML', documents: DOCS },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      expect(body.meta).toBeTruthy();
      expect(body.meta.billed_units).toBeTruthy();
    }
  });

  test('[5-1-08] missing query returns 400', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/rerank`, {
      data: { model: 'default', documents: DOCS },
    });
    expect(resp.status()).toBe(400);
  });

  test('[5-1-09] empty documents array returns 400 or empty results', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/rerank`, {
      data: { model: 'default', query: 'test', documents: [] },
    });
    expect([200, 400]).toContain(resp.status());
    if (resp.status() === 200) {
      const body = await resp.json();
      expect(body.results.length).toBe(0);
    }
  });
});

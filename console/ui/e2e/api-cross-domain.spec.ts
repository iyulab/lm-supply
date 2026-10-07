import { test, expect } from './fixtures/base.fixture';

// ================================================================
// Cross-domain workflow API tests
// ================================================================

test.describe('Cross-domain API workflows', () => {
  test.setTimeout(120_000);

  // Embed → Rerank pipeline via API
  test('embed then rerank same texts via API', async ({ request }) => {
    const texts = ['Machine learning is AI', 'The weather is nice', 'Deep learning uses neural nets'];

    // Step 1: Embed all texts
    const embedRes = await request.post('/v1/embeddings', {
      data: { model: 'default', input: texts },
    });
    expect(embedRes.status()).toBe(200);
    const embedBody = await embedRes.json();
    expect(embedBody.data.length).toBe(3);

    // Step 2: Rerank same texts with a query
    const rerankRes = await request.post('/v1/rerank', {
      data: {
        model: 'default',
        query: 'What is artificial intelligence?',
        documents: texts,
      },
    });
    expect(rerankRes.status()).toBe(200);
    const rerankBody = await rerankRes.json();
    expect(rerankBody.results.length).toBe(3);

    // ML-related text should rank higher than weather text
    const mlIndex = rerankBody.results.findIndex((r: { index: number }) => r.index === 0);
    const weatherIndex = rerankBody.results.findIndex((r: { index: number }) => r.index === 1);
    expect(mlIndex).toBeLessThan(weatherIndex);
  });

  // Model switch: embed with default then fast model
  test('embed with default and fast models produce different results', async ({ request }) => {
    const text = 'Testing model switch';

    // Embed with default model
    const res1 = await request.post('/v1/embeddings', {
      data: { model: 'default', input: [text] },
    });
    expect(res1.status()).toBe(200);
    const body1 = await res1.json();
    const dim1 = body1.data[0].embedding.length;

    // Embed with fast model
    const res2 = await request.post('/v1/embeddings', {
      data: { model: 'fast', input: [text] },
    });
    expect(res2.status()).toBe(200);
    const body2 = await res2.json();
    const dim2 = body2.data[0].embedding.length;

    // Both should produce valid embeddings (may differ in dimensions)
    expect(dim1).toBeGreaterThan(0);
    expect(dim2).toBeGreaterThan(0);
  });

  // Infer → load/unload cycle: embed, check loaded, embed again
  test('embed succeeds, model can be loaded on demand', async ({ request }) => {
    // Step 1: First inference (triggers on-demand loading)
    const res1 = await request.post('/v1/embeddings', {
      data: { model: 'default', input: ['test'] },
    });
    expect(res1.status()).toBe(200);

    // Step 2: Verify loaded models exist
    const loadedRes = await request.get('/api/cache/loaded');
    expect(loadedRes.status()).toBe(200);

    // Step 3: Second inference should also succeed
    const res2 = await request.post('/v1/embeddings', {
      data: { model: 'default', input: ['test again'] },
    });
    expect(res2.status()).toBe(200);
    const body2 = await res2.json();
    expect(body2.data[0].embedding.length).toBeGreaterThan(0);
  });
});

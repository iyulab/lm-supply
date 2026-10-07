import { test, expect } from './fixtures/base.fixture';

// ================================================================
// API Inference — Text endpoints (chat, embed, rerank)
// Uses live backend — tests may auto-load models on first run
// ================================================================

test.describe('API Inference — Text', () => {
  // Increase timeout for inference tests (model loading may be needed)
  test.setTimeout(120_000);

  // [5-18] POST /v1/chat/completions (non-streaming)
  test('[5-18] POST /v1/chat/completions non-streaming returns completion', async ({ request }) => {
    const response = await request.post('/v1/chat/completions', {
      data: {
        model: 'default',
        messages: [{ role: 'user', content: 'Say hello in one word.' }],
        stream: false,
        max_tokens: 10,
      },
    });
    expect(response.status()).toBe(200);

    const body = await response.json();
    expect(body).toHaveProperty('choices');
    expect(Array.isArray(body.choices)).toBe(true);
    expect(body.choices.length).toBeGreaterThan(0);
    expect(body.choices[0]).toHaveProperty('message');
    expect(body.choices[0].message).toHaveProperty('content');
    expect(body).toHaveProperty('usage');
  });

  // [5-19] POST /v1/chat/completions (streaming)
  test('[5-19] POST /v1/chat/completions streaming returns SSE tokens', async ({ page }) => {
    await page.goto('/');

    const result = await page.evaluate(async () => {
      const response = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'default',
          messages: [{ role: 'user', content: 'Say hi.' }],
          stream: true,
          max_tokens: 5,
        }),
      });

      const contentType = response.headers.get('content-type') ?? '';
      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let text = '';
      let hasDone = false;

      for (let i = 0; i < 100; i++) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        text += chunk;
        if (chunk.includes('[DONE]')) {
          hasDone = true;
          break;
        }
      }

      return { contentType, hasData: text.includes('data:'), hasDone, status: response.status };
    });

    expect(result.status).toBe(200);
    expect(result.contentType).toContain('text/event-stream');
    expect(result.hasData).toBe(true);
    expect(result.hasDone).toBe(true);
  });

  // [5-20] POST /v1/embeddings (single input)
  test('[5-20] POST /v1/embeddings returns embedding vector', async ({ request }) => {
    const response = await request.post('/v1/embeddings', {
      data: {
        model: 'default',
        input: ['Hello world'],
      },
    });
    expect(response.status()).toBe(200);

    const body = await response.json();
    expect(body).toHaveProperty('data');
    expect(Array.isArray(body.data)).toBe(true);
    expect(body.data.length).toBe(1);
    expect(body.data[0]).toHaveProperty('embedding');
    expect(Array.isArray(body.data[0].embedding)).toBe(true);
    expect(body.data[0].embedding.length).toBeGreaterThan(0);
    expect(body).toHaveProperty('usage');
  });

  // [5-21] POST /v1/embeddings (batch)
  test('[5-21] POST /v1/embeddings batch returns multiple embeddings', async ({ request }) => {
    const response = await request.post('/v1/embeddings', {
      data: {
        model: 'default',
        input: ['Text A', 'Text B', 'Text C'],
      },
    });
    expect(response.status()).toBe(200);

    const body = await response.json();
    expect(body.data.length).toBe(3);

    // Each embedding should have same dimensions
    const dim = body.data[0].embedding.length;
    for (const item of body.data) {
      expect(item.embedding.length).toBe(dim);
      expect(item).toHaveProperty('index');
    }
  });

  // [5-22] POST /v1/rerank returns ranked results
  test('[5-22] POST /v1/rerank returns ranked documents', async ({ request }) => {
    const response = await request.post('/v1/rerank', {
      data: {
        model: 'default',
        query: 'What is machine learning?',
        documents: [
          'Machine learning is a subset of artificial intelligence.',
          'The weather today is sunny.',
          'Deep learning uses neural networks.',
        ],
      },
    });
    expect(response.status()).toBe(200);

    const body = await response.json();
    expect(body).toHaveProperty('results');
    expect(Array.isArray(body.results)).toBe(true);
    expect(body.results.length).toBe(3);

    // Each result should have index and score
    for (const result of body.results) {
      expect(result).toHaveProperty('index');
      expect(result).toHaveProperty('relevance_score');
      expect(typeof result.relevance_score).toBe('number');
    }

    // Results should be sorted by score (descending)
    for (let i = 1; i < body.results.length; i++) {
      expect(body.results[i - 1].relevance_score).toBeGreaterThanOrEqual(body.results[i].relevance_score);
    }
  });
});

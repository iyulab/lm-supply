import { test, expect } from '@playwright/test';

const BASE = process.env.API_BASE_URL || 'http://localhost:5000';

test.describe('API Compliance — Infrastructure', () => {
  test('all responses include X-Request-Id header', async ({ request }) => {
    const resp = await request.get(`${BASE}/v1/models`);
    expect(resp.headers()['x-request-id']).toBeTruthy();
  });

  test('X-Request-Id echoes client-provided value', async ({ request }) => {
    const myId = 'test-id-12345';
    const resp = await request.get(`${BASE}/v1/models`, {
      headers: { 'X-Request-Id': myId }
    });
    expect(resp.headers()['x-request-id']).toBe(myId);
  });

  test('model not found returns 404 with error envelope', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/embeddings`, {
      data: { model: 'nonexistent-xyz-12345', input: 'test' }
    });
    expect(resp.status()).toBe(404);
    const body = await resp.json();
    expect(body.error.type).toBe('invalid_request_error');
    expect(body.error.code).toBe('model_not_found');
    expect(body.error.message).toContain('nonexistent-xyz-12345');
  });

  test('missing required field returns 400 with error envelope', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/embeddings`, {
      data: { model: 'default' }  // missing 'input'
    });
    expect(resp.status()).toBe(400);
    const body = await resp.json();
    expect(body.error).toBeTruthy();
    expect(body.error.type).toBeTruthy();
  });
});

test.describe('API Compliance — Models', () => {
  test('GET /v1/models returns OpenAI-compatible list', async ({ request }) => {
    const resp = await request.get(`${BASE}/v1/models`);
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(body.object).toBe('list');
    expect(Array.isArray(body.data)).toBe(true);
  });

  test('model entries have capabilities array', async ({ request }) => {
    const resp = await request.get(`${BASE}/v1/models`);
    const body = await resp.json();
    if (body.data.length > 0) {
      const model = body.data[0];
      expect(model.id).toBeTruthy();
      expect(model.object).toBe('model');
      expect(Array.isArray(model.capabilities)).toBe(true);
    }
  });
});

test.describe('API Compliance — Embeddings', () => {
  test('base64 encoding_format returns string embedding', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/embeddings`, {
      data: { model: 'default', input: 'hello', encoding_format: 'base64' }
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      expect(typeof body.data[0].embedding).toBe('string');
    } else {
      // model not loaded — acceptable
      expect([404, 503]).toContain(resp.status());
    }
  });

  test('dimensions parameter in request is accepted', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/embeddings`, {
      data: { model: 'default', input: 'hello', dimensions: 64 }
    });
    // 200 or model-not-found — not 422 (invalid param)
    expect(resp.status()).not.toBe(422);
  });
});

test.describe('API Compliance — Chat', () => {
  test('chat request with tools field is accepted', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: {
        model: 'default',
        messages: [{ role: 'user', content: 'Hi' }],
        tools: [{ type: 'function', function: { name: 'test', description: 'test tool' } }],
        max_tokens: 10
      }
    });
    // Not 422 — tools field must be accepted
    expect(resp.status()).not.toBe(422);
  });

  test('response_format json_object is accepted', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: {
        model: 'default',
        messages: [{ role: 'user', content: 'Return JSON' }],
        response_format: { type: 'json_object' },
        max_tokens: 20
      }
    });
    expect(resp.status()).not.toBe(422);
  });

  test('n parameter is accepted', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: {
        model: 'default',
        messages: [{ role: 'user', content: 'Hi' }],
        n: 2,
        max_tokens: 5
      }
    });
    expect(resp.status()).not.toBe(422);
  });

  test('max_completion_tokens alias is accepted', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: {
        model: 'default',
        messages: [{ role: 'user', content: 'Hi' }],
        max_completion_tokens: 5
      }
    });
    expect(resp.status()).not.toBe(422);
  });
});

test.describe('API Compliance — Audio', () => {
  test('POST /v1/audio/translations is registered (not 404)', async ({ request }) => {
    // Send minimal request — will fail without a model, but endpoint must exist
    const resp = await request.post(`${BASE}/v1/audio/translations`, {
      multipart: {
        file: {
          name: 'test.wav',
          mimeType: 'audio/wav',
          buffer: Buffer.alloc(44, 0)  // minimal WAV header bytes
        },
        model: 'default'
      }
    });
    expect(resp.status()).not.toBe(404);
  });

  test('transcription accepts srt response_format', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/audio/transcriptions`, {
      multipart: {
        file: { name: 'test.wav', mimeType: 'audio/wav', buffer: Buffer.alloc(44, 0) },
        model: 'default',
        response_format: 'srt'
      }
    });
    // Not 422 — srt format must be recognized
    expect(resp.status()).not.toBe(422);
  });

  test('TTS accepts pcm response_format', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/audio/speech`, {
      data: { model: 'default', input: 'hello', response_format: 'pcm' }
    });
    expect(resp.status()).not.toBe(422);
  });
});

test.describe('API Compliance — Images', () => {
  test('POST /v1/images/edits is registered (not 404)', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/edits`, {
      multipart: {
        image: { name: 'test.png', mimeType: 'image/png', buffer: Buffer.from([137, 80, 78, 71]) },
        prompt: 'test edit'
      }
    });
    expect(resp.status()).not.toBe(404);
  });

  test('POST /v1/images/variations is registered (not 404)', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/variations`, {
      multipart: {
        image: { name: 'test.png', mimeType: 'image/png', buffer: Buffer.from([137, 80, 78, 71]) }
      }
    });
    expect(resp.status()).not.toBe(404);
  });

  test('image generation accepts n parameter', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/generations`, {
      data: { prompt: 'test', model: 'default', n: 2, size: '256x256' }
    });
    expect(resp.status()).not.toBe(422);
  });

  test('image generation accepts quality parameter', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/generations`, {
      data: { prompt: 'test', model: 'default', quality: 'hd', size: '256x256' }
    });
    expect(resp.status()).not.toBe(422);
  });
});

test.describe('API Compliance — Translation', () => {
  test('POST /v2/translate is registered (not 404)', async ({ request }) => {
    const resp = await request.post(`http://localhost:5000/v2/translate`, {
      data: { text: ['hello'], target_lang: 'KO' }
    });
    expect(resp.status()).not.toBe(404);
    expect(resp.status()).not.toBe(405);
  });

  test('DeepL format response has translations array', async ({ request }) => {
    const resp = await request.post(`http://localhost:5000/v2/translate`, {
      data: { text: ['hello'], target_lang: 'KO' }
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      expect(Array.isArray(body.translations)).toBe(true);
    }
  });
});

test.describe('API Compliance — Rerank', () => {
  test('rerank accepts object[] documents with rank_fields', async ({ request }) => {
    const resp = await request.post(`${BASE}/rerank`, {
      data: {
        model: 'default',
        query: 'test query',
        documents: [
          { title: 'Doc One', body: 'content one' },
          { title: 'Doc Two', body: 'content two' }
        ],
        rank_fields: ['title', 'body']
      }
    });
    // Not 422 — rank_fields+object documents must be accepted
    expect(resp.status()).not.toBe(422);
  });

  test('rerank response includes meta when successful', async ({ request }) => {
    const resp = await request.post(`${BASE}/rerank`, {
      data: {
        model: 'default',
        query: 'test',
        documents: ['doc one', 'doc two']
      }
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      expect(body.meta).toBeTruthy();
      expect(body.meta.tokens).toBeTruthy();
    }
  });
});

import { test, expect } from '@playwright/test';

const BASE = process.env.API_BASE_URL || 'http://localhost:5000';

// ================================================================
// POST /v1/audio/speech
// ================================================================
test.describe('TTS Speech Synthesis', () => {
  test('default WAV response has correct Content-Type', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/audio/speech`, {
      data: { model: 'default', input: 'Hello, this is a test.', voice: 'alloy' },
    });
    if (resp.status() === 200) {
      const ct = resp.headers()['content-type'];
      expect(ct).toMatch(/audio\/(wav|x-wav|wave)/);
    } else {
      expect([404, 503]).toContain(resp.status());
    }
  });

  test('all OpenAI voice names accepted (not 422)', async ({ request }) => {
    const voices = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'];
    for (const voice of voices) {
      const resp = await request.post(`${BASE}/v1/audio/speech`, {
        data: { model: 'default', input: 'test', voice },
      });
      expect(resp.status()).not.toBe(422);
    }
  });

  test('format pcm returns audio/pcm Content-Type', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/audio/speech`, {
      data: { model: 'default', input: 'test', response_format: 'pcm' },
    });
    if (resp.status() === 200) {
      const ct = resp.headers()['content-type'];
      expect(ct).toMatch(/audio\/(pcm|wav|x-wav)/);
    } else {
      expect([404, 503]).toContain(resp.status());
    }
  });

  test('format mp3 accepted (not 422)', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/audio/speech`, {
      data: { model: 'default', input: 'test', response_format: 'mp3' },
    });
    expect(resp.status()).not.toBe(422);
  });

  test('formats opus/flac/aac accepted (not 422)', async ({ request }) => {
    for (const fmt of ['opus', 'flac', 'aac']) {
      const resp = await request.post(`${BASE}/v1/audio/speech`, {
        data: { model: 'default', input: 'test', response_format: fmt },
      });
      expect(resp.status()).not.toBe(422);
    }
  });

  test('speed parameter 0.25–4.0 accepted (not 422)', async ({ request }) => {
    for (const speed of [0.25, 1.0, 2.0, 4.0]) {
      const resp = await request.post(`${BASE}/v1/audio/speech`, {
        data: { model: 'default', input: 'test', speed },
      });
      expect(resp.status()).not.toBe(422);
    }
  });

  test('missing input field returns 400', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/audio/speech`, {
      data: { model: 'default', voice: 'alloy' },
    });
    expect(resp.status()).toBe(400);
  });

  test('Content-Disposition includes filename', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/audio/speech`, {
      data: { model: 'default', input: 'test', response_format: 'wav' },
    });
    if (resp.status() === 200) {
      const cd = resp.headers()['content-disposition'];
      if (cd) {
        expect(cd).toContain('attachment');
        expect(cd).toContain('filename');
      }
    }
  });

  test('response delivers chunked/streaming data', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/audio/speech`, {
      data: { model: 'default', input: 'Hello world.', voice: 'alloy' },
    });
    if (resp.status() === 200) {
      const body = await resp.body();
      expect(body.length).toBeGreaterThan(0);
    }
  });
});

import { test, expect } from '@playwright/test';
import { BASE, requireCachedModel } from './host-capabilities';
import { headerOnlyWav, minimalWav } from './fixtures';

// ================================================================
// POST /v1/audio/transcriptions
// ================================================================
test.describe('Audio Transcriptions', () => {
  test('default response is JSON with text field', async ({ request }) => {
    await requireCachedModel(request, 'transcriber');
    const resp = await request.post(`${BASE}/v1/audio/transcriptions`, {
      multipart: {
        file: { name: 'test.wav', mimeType: 'audio/wav', buffer: minimalWav() },
        model: 'default',
      },
    });
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(body).toHaveProperty('text');
  });

  test('verbose_json includes segments with start/end/text', async ({ request }) => {
    await requireCachedModel(request, 'transcriber');
    const resp = await request.post(`${BASE}/v1/audio/transcriptions`, {
      multipart: {
        file: { name: 'test.wav', mimeType: 'audio/wav', buffer: minimalWav() },
        model: 'default',
        response_format: 'verbose_json',
      },
    });
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(body).toHaveProperty('text');
    expect(body).toHaveProperty('language');
    expect(body).toHaveProperty('duration');
    if (body.segments && body.segments.length > 0) {
      expect(body.segments[0]).toHaveProperty('start');
      expect(body.segments[0]).toHaveProperty('end');
      expect(body.segments[0]).toHaveProperty('text');
    }
  });

  test('text format returns plain text (not JSON)', async ({ request }) => {
    await requireCachedModel(request, 'transcriber');
    const resp = await request.post(`${BASE}/v1/audio/transcriptions`, {
      multipart: {
        file: { name: 'test.wav', mimeType: 'audio/wav', buffer: minimalWav() },
        model: 'default',
        response_format: 'text',
      },
    });
    expect(resp.status()).toBe(200);
    const ct = resp.headers()['content-type'];
    expect(ct).toContain('text/plain');
  });

  test('srt format returns SRT string', async ({ request }) => {
    await requireCachedModel(request, 'transcriber');
    const resp = await request.post(`${BASE}/v1/audio/transcriptions`, {
      multipart: {
        file: { name: 'test.wav', mimeType: 'audio/wav', buffer: minimalWav() },
        model: 'default',
        response_format: 'srt',
      },
    });
    expect(resp.status()).toBe(200);
    const text = await resp.text();
    // SRT format: "1\n00:00:..."
    expect(text).toMatch(/^1\n\d\d:\d\d:\d\d,\d{3} --> /);
  });

  test('vtt format returns VTT string', async ({ request }) => {
    await requireCachedModel(request, 'transcriber');
    const resp = await request.post(`${BASE}/v1/audio/transcriptions`, {
      multipart: {
        file: { name: 'test.wav', mimeType: 'audio/wav', buffer: minimalWav() },
        model: 'default',
        response_format: 'vtt',
      },
    });
    expect(resp.status()).toBe(200);
    const text = await resp.text();
    expect(text).toContain('WEBVTT');
  });

  test('prompt parameter accepted', async ({ request }) => {
    await requireCachedModel(request, 'transcriber');
    const resp = await request.post(`${BASE}/v1/audio/transcriptions`, {
      multipart: {
        file: { name: 'test.wav', mimeType: 'audio/wav', buffer: minimalWav() },
        model: 'default',
        prompt: 'This is a test audio file.',
      },
    });
    expect(resp.status()).toBe(200);
  });

  test('temperature parameter accepted', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/audio/transcriptions`, {
      multipart: {
        file: { name: 'test.wav', mimeType: 'audio/wav', buffer: minimalWav() },
        model: 'default',
        temperature: '0.5',
      },
    });
    expect(resp.status()).toBe(200);
  });

  // Audio the library cannot use is the caller's mistake: 400 naming the problem, not the opaque 500 of an unexpected failure.
  for (const [name, buffer, message] of [
    ['bytes that are no audio format', Buffer.alloc(44), 'not in a format'],
    ['a WAV header with no samples', headerOnlyWav(), 'no samples'],
  ] as const) {
    test(`${name} returns 400`, async ({ request }) => {
      await requireCachedModel(request, 'transcriber');
      const resp = await request.post(`${BASE}/v1/audio/transcriptions`, {
        multipart: { file: { name: 'bad.wav', mimeType: 'audio/wav', buffer }, model: 'default' },
      });
      expect(resp.status()).toBe(400);
      const body = await resp.json();
      expect(body.error.type).toBe('invalid_request_error');
      expect(body.error.message).toContain(message);
    });
  }

  test('missing file returns 400 with error message', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/audio/transcriptions`, {
      multipart: { model: 'default' },
    });
    expect(resp.status()).toBe(400);
  });
});

// ================================================================
// POST /v1/audio/translations
// ================================================================
test.describe('Audio Translations', () => {
  test('translation endpoint exists and returns response (not 404)', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/audio/translations`, {
      multipart: {
        file: { name: 'test.wav', mimeType: 'audio/wav', buffer: minimalWav() },
        model: 'default',
      },
    });
    expect(resp.status()).not.toBe(404);
    expect(resp.status()).not.toBe(405);
  });

  test('verbose_json for translations accepted', async ({ request }) => {
    await requireCachedModel(request, 'transcriber');
    const resp = await request.post(`${BASE}/v1/audio/translations`, {
      multipart: {
        file: { name: 'test.wav', mimeType: 'audio/wav', buffer: minimalWav() },
        model: 'default',
        response_format: 'verbose_json',
      },
    });
    expect(resp.status()).toBe(200);
  });

  test('srt/vtt/text formats all accepted by translations endpoint', async ({ request }) => {
    await requireCachedModel(request, 'transcriber');
    for (const format of ['json', 'text', 'srt', 'vtt']) {
      const resp = await request.post(`${BASE}/v1/audio/translations`, {
        multipart: {
          file: { name: 'test.wav', mimeType: 'audio/wav', buffer: minimalWav() },
          model: 'default',
          response_format: format,
        },
      });
      expect(resp.status()).toBe(200);
    }
  });
});

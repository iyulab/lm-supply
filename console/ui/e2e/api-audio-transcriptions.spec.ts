import { test, expect } from '@playwright/test';

const BASE = process.env.API_BASE_URL || 'http://localhost:5000';

/** Minimal valid WAV buffer (44-byte header, 1 second silence at 8kHz) */
function minimalWav(): Buffer {
  const sampleRate = 8000;
  const numSamples = sampleRate;
  const dataSize = numSamples * 2;
  const buf = Buffer.alloc(44 + dataSize);
  buf.write('RIFF', 0); buf.writeUInt32LE(36 + dataSize, 4); buf.write('WAVE', 8);
  buf.write('fmt ', 12); buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20);
  buf.writeUInt16LE(1, 22); buf.writeUInt32LE(sampleRate, 24);
  buf.writeUInt32LE(sampleRate * 2, 28); buf.writeUInt16LE(2, 32); buf.writeUInt16LE(16, 34);
  buf.write('data', 36); buf.writeUInt32LE(dataSize, 40);
  return buf;
}

// ================================================================
// POST /v1/audio/transcriptions
// ================================================================
test.describe('Audio Transcriptions', () => {
  test('default response is JSON with text field', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/audio/transcriptions`, {
      multipart: {
        file: { name: 'test.wav', mimeType: 'audio/wav', buffer: minimalWav() },
        model: 'default',
      },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      expect(body).toHaveProperty('text');
    } else {
      expect([404, 503]).toContain(resp.status());
    }
  });

  test('verbose_json includes segments with start/end/text', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/audio/transcriptions`, {
      multipart: {
        file: { name: 'test.wav', mimeType: 'audio/wav', buffer: minimalWav() },
        model: 'default',
        response_format: 'verbose_json',
      },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      expect(body).toHaveProperty('text');
      expect(body).toHaveProperty('language');
      expect(body).toHaveProperty('duration');
      if (body.segments && body.segments.length > 0) {
        expect(body.segments[0]).toHaveProperty('start');
        expect(body.segments[0]).toHaveProperty('end');
        expect(body.segments[0]).toHaveProperty('text');
      }
    }
  });

  test('text format returns plain text (not JSON)', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/audio/transcriptions`, {
      multipart: {
        file: { name: 'test.wav', mimeType: 'audio/wav', buffer: minimalWav() },
        model: 'default',
        response_format: 'text',
      },
    });
    if (resp.status() === 200) {
      const ct = resp.headers()['content-type'];
      expect(ct).toContain('text/plain');
    } else {
      expect([404, 503]).toContain(resp.status());
    }
  });

  test('srt format returns SRT string (not 422)', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/audio/transcriptions`, {
      multipart: {
        file: { name: 'test.wav', mimeType: 'audio/wav', buffer: minimalWav() },
        model: 'default',
        response_format: 'srt',
      },
    });
    expect(resp.status()).not.toBe(422);
    if (resp.status() === 200) {
      const text = await resp.text();
      // SRT format: "1\n00:00:..."
      expect(text.length).toBeGreaterThan(0);
    }
  });

  test('vtt format returns VTT string (not 422)', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/audio/transcriptions`, {
      multipart: {
        file: { name: 'test.wav', mimeType: 'audio/wav', buffer: minimalWav() },
        model: 'default',
        response_format: 'vtt',
      },
    });
    expect(resp.status()).not.toBe(422);
    if (resp.status() === 200) {
      const text = await resp.text();
      expect(text).toContain('WEBVTT');
    }
  });

  test('prompt parameter accepted (not 422)', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/audio/transcriptions`, {
      multipart: {
        file: { name: 'test.wav', mimeType: 'audio/wav', buffer: minimalWav() },
        model: 'default',
        prompt: 'This is a test audio file.',
      },
    });
    expect(resp.status()).not.toBe(422);
  });

  test('temperature parameter accepted (not 422)', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/audio/transcriptions`, {
      multipart: {
        file: { name: 'test.wav', mimeType: 'audio/wav', buffer: minimalWav() },
        model: 'default',
        temperature: '0.5',
      },
    });
    expect(resp.status()).not.toBe(422);
  });

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

  test('verbose_json for translations accepted (not 422)', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/audio/translations`, {
      multipart: {
        file: { name: 'test.wav', mimeType: 'audio/wav', buffer: minimalWav() },
        model: 'default',
        response_format: 'verbose_json',
      },
    });
    expect(resp.status()).not.toBe(422);
  });

  test('srt/vtt/text formats all accepted by translations endpoint', async ({ request }) => {
    for (const format of ['json', 'text', 'srt', 'vtt']) {
      const resp = await request.post(`${BASE}/v1/audio/translations`, {
        multipart: {
          file: { name: 'test.wav', mimeType: 'audio/wav', buffer: minimalWav() },
          model: 'default',
          response_format: format,
        },
      });
      expect(resp.status()).not.toBe(422);
    }
  });
});

import { test, expect } from './fixtures/base.fixture';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const TEST_AUDIO = path.resolve(__dirname, 'fixtures/test-files/test-audio.wav');
const TEST_IMAGE = path.resolve(__dirname, 'fixtures/test-files/test-image.png');

// ================================================================
// API Inference — Audio + Media endpoints
// ================================================================

test.describe('API Inference — Audio & Media', () => {
  test.setTimeout(120_000);

  // [5-23] POST /v1/audio/transcriptions returns transcription text
  test('[5-23] POST /v1/audio/transcriptions returns text', async ({ request }) => {
    const audioBuffer = fs.readFileSync(TEST_AUDIO);
    const response = await request.post('/v1/audio/transcriptions', {
      multipart: {
        model: 'default',
        file: { name: 'test-audio.wav', mimeType: 'audio/wav', buffer: audioBuffer },
      },
    });
    expect(response.status()).toBe(200);

    const body = await response.json();
    expect(body).toHaveProperty('text');
    expect(typeof body.text).toBe('string');
  });

  // [5-24] POST /v1/audio/transcriptions with verbose_json
  test('[5-24] POST /v1/audio/transcriptions verbose returns segments', async ({ request }) => {
    const audioBuffer = fs.readFileSync(TEST_AUDIO);
    const response = await request.post('/v1/audio/transcriptions', {
      multipart: {
        model: 'default',
        file: { name: 'test-audio.wav', mimeType: 'audio/wav', buffer: audioBuffer },
        response_format: 'verbose_json',
      },
    });
    expect(response.status()).toBe(200);

    const body = await response.json();
    expect(body).toHaveProperty('text');
    expect(body).toHaveProperty('language');
    expect(body).toHaveProperty('duration');
    expect(body).toHaveProperty('segments');
    expect(Array.isArray(body.segments)).toBe(true);
  });

  // [5-25] POST /v1/audio/speech returns audio or handled error
  test('[5-25] POST /v1/audio/speech returns audio or error', async ({ request }) => {
    const response = await request.post('/v1/audio/speech', {
      data: {
        model: 'default',
        input: 'Hello',
      },
    });

    // Should return 200 with audio or a handled error (not a crash)
    if (response.status() === 200) {
      const contentType = response.headers()['content-type'] ?? '';
      expect(contentType).toContain('audio/');
      const body = await response.body();
      expect(body.length).toBeGreaterThan(0);
    } else {
      // Model may not be available — verify error is structured
      const body = await response.json();
      expect(body).toHaveProperty('error');
      expect(body.error).toHaveProperty('message');
    }
  });

  // [5-26] POST /v1/translate returns translation
  test('[5-26] POST /v1/translate returns translated text', async ({ request }) => {
    const response = await request.post('/v1/translate', {
      data: {
        model: 'default',
        input: 'Hello world',
        source: 'en',
        target: 'ko',
      },
    });
    expect(response.status()).toBe(200);

    const body = await response.json();
    expect(body).toHaveProperty('translations');
    expect(Array.isArray(body.translations)).toBe(true);
    expect(body.translations.length).toBeGreaterThan(0);
    expect(body.translations[0]).toHaveProperty('translated_text');
    expect(body.translations[0]).toHaveProperty('source_text');
  });

  // [5-27] POST /v1/images/caption returns caption or handled error
  test('[5-27] POST /v1/images/caption returns caption or error', async ({ request }) => {
    const imageBuffer = fs.readFileSync(TEST_IMAGE);
    const response = await request.post('/v1/images/caption', {
      multipart: {
        model: 'default',
        file: { name: 'test-image.png', mimeType: 'image/png', buffer: imageBuffer },
      },
    });

    if (response.status() === 200) {
      const body = await response.json();
      expect(body).toHaveProperty('caption');
      expect(typeof body.caption).toBe('string');
      expect(body.caption.length).toBeGreaterThan(0);
    } else {
      const body = await response.json();
      expect(body).toHaveProperty('error');
      expect(body.error).toHaveProperty('message');
    }
  });

  // [5-28] POST /v1/images/vqa returns answer or handled error
  test('[5-28] POST /v1/images/vqa returns answer or error', async ({ request }) => {
    const imageBuffer = fs.readFileSync(TEST_IMAGE);
    const response = await request.post('/v1/images/vqa', {
      multipart: {
        model: 'default',
        file: { name: 'test-image.png', mimeType: 'image/png', buffer: imageBuffer },
        question: 'What is in this image?',
      },
    });

    if (response.status() === 200) {
      const body = await response.json();
      expect(body).toHaveProperty('answer');
      expect(typeof body.answer).toBe('string');
    } else {
      const body = await response.json();
      expect(body).toHaveProperty('error');
      expect(body.error).toHaveProperty('message');
    }
  });

  // [5-29] POST /v1/images/ocr returns text or handled error
  test('[5-29] POST /v1/images/ocr returns detected text or error', async ({ request }) => {
    const imageBuffer = fs.readFileSync(TEST_IMAGE);
    const response = await request.post('/v1/images/ocr', {
      multipart: {
        model: 'default',
        file: { name: 'test-image.png', mimeType: 'image/png', buffer: imageBuffer },
      },
    });

    if (response.status() === 200) {
      const body = await response.json();
      expect(body).toHaveProperty('text');
      expect(body).toHaveProperty('blocks');
      expect(Array.isArray(body.blocks)).toBe(true);
    } else {
      const body = await response.json();
      expect(body).toHaveProperty('error');
      expect(body.error).toHaveProperty('message');
    }
  });

  // [5-30] GET /v1/images/ocr/languages returns language list
  test('[5-30] GET /v1/images/ocr/languages returns language codes', async ({ request }) => {
    const response = await request.get('/v1/images/ocr/languages');
    expect(response.status()).toBe(200);

    const body = await response.json();
    expect(body).toHaveProperty('languages');
    expect(Array.isArray(body.languages)).toBe(true);
    expect(body.languages.length).toBeGreaterThan(10);
    expect(body.languages).toContain('en');
    expect(body.languages).toContain('ko');
  });

  // [5-31] POST /v1/images/detect returns objects or handled error
  test('[5-31] POST /v1/images/detect returns objects or error', async ({ request }) => {
    const imageBuffer = fs.readFileSync(TEST_IMAGE);
    const response = await request.post('/v1/images/detect', {
      multipart: {
        model: 'default',
        file: { name: 'test-image.png', mimeType: 'image/png', buffer: imageBuffer },
      },
    });

    if (response.status() === 200) {
      const body = await response.json();
      expect(body).toHaveProperty('objects');
      expect(Array.isArray(body.objects)).toBe(true);
    } else {
      const body = await response.json();
      expect(body).toHaveProperty('error');
      expect(body.error).toHaveProperty('message');
    }
  });

  // [5-32] GET /v1/images/detect/labels returns COCO labels
  test('[5-32] GET /v1/images/detect/labels returns 80 COCO labels', async ({ request }) => {
    const response = await request.get('/v1/images/detect/labels');
    expect(response.status()).toBe(200);

    const body = await response.json();
    expect(body).toHaveProperty('labels');
    expect(Array.isArray(body.labels)).toBe(true);
    expect(body.labels.length).toBe(80);
    expect(body.labels[0]).toHaveProperty('id');
    expect(body.labels[0]).toHaveProperty('name');
    expect(body.labels[0].name).toBe('person');
  });

  // [5-33] POST /v1/images/segment returns segments or handled error
  test('[5-33] POST /v1/images/segment returns segments or error', async ({ request }) => {
    const imageBuffer = fs.readFileSync(TEST_IMAGE);
    const response = await request.post('/v1/images/segment', {
      multipart: {
        model: 'default',
        file: { name: 'test-image.png', mimeType: 'image/png', buffer: imageBuffer },
      },
    });

    if (response.status() === 200) {
      const body = await response.json();
      expect(body).toHaveProperty('segments');
      expect(Array.isArray(body.segments)).toBe(true);
      if (body.segments.length > 0) {
        expect(body.segments[0]).toHaveProperty('label');
        expect(body.segments[0]).toHaveProperty('score');
      }
    } else {
      const body = await response.json();
      expect(body).toHaveProperty('error');
      expect(body.error).toHaveProperty('message');
    }
  });

  // [5-35] GET /v1/images/segment/labels returns ADE20K labels
  test('[5-35] GET /v1/images/segment/labels returns 150 ADE20K labels', async ({ request }) => {
    const response = await request.get('/v1/images/segment/labels');
    expect(response.status()).toBe(200);

    const body = await response.json();
    expect(body).toHaveProperty('labels');
    expect(Array.isArray(body.labels)).toBe(true);
    expect(body.labels.length).toBe(150);
    expect(body.labels[0]).toHaveProperty('id');
    expect(body.labels[0]).toHaveProperty('name');
  });
});

import { test, expect } from './fixtures/base.fixture';
import { BASE, requireCachedModel, requireCachedRepo } from './host-capabilities';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const TEST_AUDIO = path.resolve(__dirname, 'fixtures/test-files/test-audio.wav');
const TEST_IMAGE = path.resolve(__dirname, 'fixtures/test-files/test-image.png');

/** The repository that holds the OCR detection and recognition models. */
const OCR_REPO = 'monkt/paddleocr-onnx';

// ================================================================
// API Inference — Audio + Media endpoints
// ================================================================

test.describe('API Inference — Audio & Media', () => {
  test.setTimeout(120_000);

  // POST /v1/audio/transcriptions returns transcription text
  test('POST /v1/audio/transcriptions returns text', async ({ request }) => {
    const audioBuffer = fs.readFileSync(TEST_AUDIO);
    const response = await request.post(`${BASE}/v1/audio/transcriptions`, {
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

  // POST /v1/audio/transcriptions with verbose_json
  test('POST /v1/audio/transcriptions verbose returns segments', async ({ request }) => {
    const audioBuffer = fs.readFileSync(TEST_AUDIO);
    const response = await request.post(`${BASE}/v1/audio/transcriptions`, {
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

  // POST /v1/audio/speech returns audio
  test('POST /v1/audio/speech returns audio', async ({ request }) => {
    await requireCachedModel(request, 'synthesizer');
    const response = await request.post(`${BASE}/v1/audio/speech`, {
      data: {
        model: 'default',
        input: 'Hello',
      },
    });

    expect(response.status()).toBe(200);
    const contentType = response.headers()['content-type'] ?? '';
    expect(contentType).toContain('audio/');
    const body = await response.body();
    expect(body.length).toBeGreaterThan(0);
  });

  // POST /v1/translate returns translation
  test('POST /v1/translate returns translated text', async ({ request }) => {
    const response = await request.post(`${BASE}/v1/translate`, {
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

  // POST /v1/images/caption returns a caption
  test('POST /v1/images/caption returns a caption', async ({ request }) => {
    await requireCachedModel(request, 'captioner');
    const imageBuffer = fs.readFileSync(TEST_IMAGE);
    const response = await request.post(`${BASE}/v1/images/caption`, {
      multipart: {
        model: 'default',
        file: { name: 'test-image.png', mimeType: 'image/png', buffer: imageBuffer },
      },
    });

    expect(response.status()).toBe(200);
    const body = await response.json();
    expect(body.choices[0]).toHaveProperty('caption');
    expect(typeof body.choices[0].caption).toBe('string');
    expect(body.choices[0].caption.length).toBeGreaterThan(0);
  });

  // POST /v1/images/vqa: the registered captioners do not answer questions, so the request is a client error
  test('POST /v1/images/vqa with a model that does not support VQA returns 400', async ({ request }) => {
    await requireCachedModel(request, 'captioner');
    const imageBuffer = fs.readFileSync(TEST_IMAGE);
    const response = await request.post(`${BASE}/v1/images/vqa`, {
      multipart: {
        model: 'default',
        file: { name: 'test-image.png', mimeType: 'image/png', buffer: imageBuffer },
        question: 'What is in this image?',
      },
    });

    expect(response.status()).toBe(400);
    const body = await response.json();
    expect(body.error.type).toBe('invalid_request_error');
    expect(body.error.message).toContain('visual question answering');
  });

  // POST /v1/images/ocr returns the recognized text
  test('POST /v1/images/ocr returns the recognized text', async ({ request }) => {
    await requireCachedRepo(request, OCR_REPO, 'OCR models');
    const imageBuffer = fs.readFileSync(TEST_IMAGE);
    const response = await request.post(`${BASE}/v1/images/ocr`, {
      multipart: {
        model: 'default',
        file: { name: 'test-image.png', mimeType: 'image/png', buffer: imageBuffer },
      },
    });

    expect(response.status()).toBe(200);
    const body = await response.json();
    expect(typeof body.full_text).toBe('string');
    expect(Array.isArray(body.pages)).toBe(true);
  });

  // GET /v1/images/ocr/languages returns language list
  test('GET /v1/images/ocr/languages returns language codes', async ({ request }) => {
    const response = await request.get(`${BASE}/v1/images/ocr/languages`);
    expect(response.status()).toBe(200);

    const body = await response.json();
    expect(body).toHaveProperty('languages');
    expect(Array.isArray(body.languages)).toBe(true);
    expect(body.languages.length).toBeGreaterThan(10);
    expect(body.languages).toContain('en');
    expect(body.languages).toContain('ko');
  });

  // POST /v1/images/detect returns objects
  test('POST /v1/images/detect returns objects', async ({ request }) => {
    await requireCachedModel(request, 'detector');
    const imageBuffer = fs.readFileSync(TEST_IMAGE);
    const response = await request.post(`${BASE}/v1/images/detect`, {
      multipart: {
        model: 'default',
        file: { name: 'test-image.png', mimeType: 'image/png', buffer: imageBuffer },
      },
    });

    expect(response.status()).toBe(200);
    const body = await response.json();
    expect(body).toHaveProperty('objects');
    expect(Array.isArray(body.objects)).toBe(true);
  });

  // GET /v1/images/detect/labels returns COCO labels
  test('GET /v1/images/detect/labels returns 80 COCO labels', async ({ request }) => {
    const response = await request.get(`${BASE}/v1/images/detect/labels`);
    expect(response.status()).toBe(200);

    const body = await response.json();
    expect(body).toHaveProperty('labels');
    expect(Array.isArray(body.labels)).toBe(true);
    expect(body.labels.length).toBe(80);
    expect(body.labels[0]).toHaveProperty('id');
    expect(body.labels[0]).toHaveProperty('name');
    expect(body.labels[0].name).toBe('person');
  });

  // POST /v1/images/segment returns segments
  test('POST /v1/images/segment returns segments', async ({ request }) => {
    await requireCachedModel(request, 'segmenter');
    const imageBuffer = fs.readFileSync(TEST_IMAGE);
    const response = await request.post(`${BASE}/v1/images/segment`, {
      multipart: {
        model: 'default',
        file: { name: 'test-image.png', mimeType: 'image/png', buffer: imageBuffer },
      },
    });

    expect(response.status()).toBe(200);
    const body = await response.json();
    expect(body).toHaveProperty('segments');
    expect(Array.isArray(body.segments)).toBe(true);
    expect(body.segments.length).toBeGreaterThan(0);
    expect(body.segments[0]).toHaveProperty('label');
    expect(body.segments[0]).toHaveProperty('score');
  });

  // GET /v1/images/segment/labels returns ADE20K labels
  test('GET /v1/images/segment/labels returns 150 ADE20K labels', async ({ request }) => {
    const response = await request.get(`${BASE}/v1/images/segment/labels`);
    expect(response.status()).toBe(200);

    const body = await response.json();
    expect(body).toHaveProperty('labels');
    expect(Array.isArray(body.labels)).toBe(true);
    expect(body.labels.length).toBe(150);
    expect(body.labels[0]).toHaveProperty('id');
    expect(body.labels[0]).toHaveProperty('name');
  });
});

import { test, expect } from '@playwright/test';
import { PNG_BUFFER, TEXT_PAGE_PNG } from './fixtures';
import { BASE, requireCachedModel, requireCachedOcr } from './host-capabilities';

// ================================================================
// POST /v1/images/ocr
// ================================================================
test.describe('OCR', () => {
  test('response: id, model, full_text, pages[].blocks[].text/confidence/bounding_box/lines', async ({ request }) => {
    await requireCachedOcr(request, 'crnn-en-v5');
    const resp = await request.post(`${BASE}/v1/images/ocr`, {
      multipart: {
        file: { name: 'text-page.png', mimeType: 'image/png', buffer: TEXT_PAGE_PNG },
      },
    });
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(body).toHaveProperty('id');
    expect(body).toHaveProperty('model');
    expect(body.full_text).toContain('receipt for seven days');
    expect(body.pages).toHaveLength(1);
    expect(body.pages[0].width).toBe(1000);
    expect(body.pages[0].height).toBe(220);
    // An image with two printed lines: two blocks, each one line, in reading order.
    expect(body.pages[0].blocks).toHaveLength(2);
    const block = body.pages[0].blocks[0];
    expect(block.text).toContain('receipt');
    expect(block.confidence).toBeGreaterThan(0.5);
    expect(block).toHaveProperty('bounding_box');
    expect(block.lines).toHaveLength(1);
    expect(block.lines[0].text).toBe(block.text);
  });

  test('full_text is a single string', async ({ request }) => {
    await requireCachedOcr(request, 'crnn-en-v5');
    const resp = await request.post(`${BASE}/v1/images/ocr`, {
      multipart: {
        file: { name: 'text-page.png', mimeType: 'image/png', buffer: TEXT_PAGE_PNG },
      },
    });
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    // The lines of the blocks, in order, joined by a newline.
    expect(body.full_text).toBe(body.pages[0].blocks.map((b: { text: string }) => b.text).join('\n'));
  });

  test('bounding_box has x, y, width, height float fields', async ({ request }) => {
    await requireCachedOcr(request, 'crnn-en-v5');
    const resp = await request.post(`${BASE}/v1/images/ocr`, {
      multipart: {
        file: { name: 'text-page.png', mimeType: 'image/png', buffer: TEXT_PAGE_PNG },
      },
    });
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    const bb = body.pages[0].blocks[0].bounding_box;
    for (const field of ['x', 'y', 'width', 'height']) expect(typeof bb[field], field).toBe('number');
    // Inside the page, and the first line is above the second.
    expect(bb.x + bb.width).toBeLessThanOrEqual(body.pages[0].width);
    expect(bb.y + bb.height).toBeLessThanOrEqual(body.pages[0].height);
    expect(bb.y).toBeLessThan(body.pages[0].blocks[1].bounding_box.y);
  });

  test('language parameter selects the recognizer of that language', async ({ request }) => {
    await requireCachedOcr(request, 'crnn-en-v5');
    const resp = await request.post(`${BASE}/v1/images/ocr`, {
      multipart: {
        file: { name: 'text-page.png', mimeType: 'image/png', buffer: TEXT_PAGE_PNG },
        language: 'en',
      },
    });
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(body.model).toContain('crnn-en-v5');
  });

  test('response does not claim a detected language', async ({ request }) => {
    await requireCachedOcr(request, 'crnn-en-v5');
    const resp = await request.post(`${BASE}/v1/images/ocr`, {
      multipart: {
        file: { name: 'test.png', mimeType: 'image/png', buffer: PNG_BUFFER },
      },
    });
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(body.detected_languages).toBeUndefined();
  });

  test('language without a recognition model returns 400 listing the supported codes', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/ocr`, {
      multipart: {
        file: { name: 'test.png', mimeType: 'image/png', buffer: PNG_BUFFER },
        language: 'km',
      },
    });
    expect(resp.status()).toBe(400);
    const body = await resp.json();
    expect(body.error.message).toContain("'km'");
    expect(body.error.message).toContain('ko');
  });

  test('missing file returns 400', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/ocr`, {
      multipart: {},
    });
    expect(resp.status()).toBe(400);
  });
});

// ================================================================
// GET /v1/images/ocr/languages
// ================================================================
test.describe('OCR Languages', () => {
  test('GET /v1/images/ocr/languages returns languages array', async ({ request }) => {
    const resp = await request.get(`${BASE}/v1/images/ocr/languages`);
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(body).toHaveProperty('languages');
    expect(Array.isArray(body.languages)).toBe(true);
  });
});

// ================================================================
// POST /v1/images/segment
// ================================================================
test.describe('Image Segmentation', () => {
  test('response: id, model, segments[].id/label/score/mask', async ({ request }) => {
    await requireCachedModel(request, 'segmenter');
    const resp = await request.post(`${BASE}/v1/images/segment`, {
      multipart: {
        file: { name: 'test.png', mimeType: 'image/png', buffer: PNG_BUFFER },
        model: 'default',
      },
    });
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(body).toHaveProperty('id');
    expect(body).toHaveProperty('model');
    expect(Array.isArray(body.segments)).toBe(true);
    expect(body.segments.length).toBeGreaterThan(0);
    expect(body.segments[0]).toHaveProperty('id');
    expect(body.segments[0]).toHaveProperty('label');
    expect(body.segments[0]).toHaveProperty('score');
  });

  test('mask_format none returns null mask', async ({ request }) => {
    await requireCachedModel(request, 'segmenter');
    const resp = await request.post(`${BASE}/v1/images/segment`, {
      multipart: {
        file: { name: 'test.png', mimeType: 'image/png', buffer: PNG_BUFFER },
        model: 'default',
        mask_format: 'none',
      },
    });
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(body.segments.length).toBeGreaterThan(0);
    expect(body.segments[0].mask ?? null).toBeNull();
  });

  test('mask_format rle returns RLE mask with counts and size', async ({ request }) => {
    await requireCachedModel(request, 'segmenter');
    const resp = await request.post(`${BASE}/v1/images/segment`, {
      multipart: {
        file: { name: 'test.png', mimeType: 'image/png', buffer: PNG_BUFFER },
        model: 'default',
        mask_format: 'rle',
      },
    });
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(body.segments.length).toBeGreaterThan(0);
    expect(body.segments[0].mask.format).toBe('rle');
    expect(body.segments[0].mask).toHaveProperty('counts');
    expect(body.segments[0].mask).toHaveProperty('size');
  });

  test('mask_format raw returns base64 mask', async ({ request }) => {
    await requireCachedModel(request, 'segmenter');
    const resp = await request.post(`${BASE}/v1/images/segment`, {
      multipart: {
        file: { name: 'test.png', mimeType: 'image/png', buffer: PNG_BUFFER },
        model: 'default',
        mask_format: 'raw',
      },
    });
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(body.segments.length).toBeGreaterThan(0);
    expect(body.segments[0].mask.format).toBe('raw');
    expect(typeof body.segments[0].mask.counts).toBe('string');
  });

  test('segment scores are between 0.0 and 1.0', async ({ request }) => {
    await requireCachedModel(request, 'segmenter');
    const resp = await request.post(`${BASE}/v1/images/segment`, {
      multipart: {
        file: { name: 'test.png', mimeType: 'image/png', buffer: PNG_BUFFER },
        model: 'default',
      },
    });
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(body.segments.length).toBeGreaterThan(0);
    for (const seg of body.segments) {
      expect(seg.score).toBeGreaterThanOrEqual(0.0);
      expect(seg.score).toBeLessThanOrEqual(1.0);
    }
  });

  test('missing file returns 400', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/segment`, {
      multipart: { model: 'default' },
    });
    expect(resp.status()).toBe(400);
  });
});

// ================================================================
// GET /v1/images/segment/labels
// ================================================================
test.describe('Segment Labels', () => {
  test('GET /v1/images/segment/labels returns ADE20K label list', async ({ request }) => {
    const resp = await request.get(`${BASE}/v1/images/segment/labels`);
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(body).toHaveProperty('labels');
    expect(Array.isArray(body.labels)).toBe(true);
    expect(body.labels.length).toBeGreaterThan(0);
    if (body.labels.length > 0) {
      expect(body.labels[0]).toHaveProperty('id');
      expect(body.labels[0]).toHaveProperty('name');
    }
  });
});

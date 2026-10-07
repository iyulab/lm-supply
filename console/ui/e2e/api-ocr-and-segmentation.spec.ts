import { test, expect } from '@playwright/test';

const BASE = process.env.API_BASE_URL || 'http://localhost:5000';

// 1x1 red pixel PNG
const PNG_BUFFER = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg==',
  'base64'
);

// ================================================================
// POST /v1/images/ocr
// ================================================================
test.describe('OCR', () => {
  test('response: id, model, full_text, pages[].blocks[].text/confidence/bounding_box/lines', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/ocr`, {
      multipart: {
        file: { name: 'test.png', mimeType: 'image/png', buffer: PNG_BUFFER },
      },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      expect(body).toHaveProperty('id');
      expect(body).toHaveProperty('model');
      expect(body).toHaveProperty('full_text');
      expect(Array.isArray(body.pages)).toBe(true);
      if (body.pages.length > 0 && body.pages[0].blocks?.length > 0) {
        const block = body.pages[0].blocks[0];
        expect(block).toHaveProperty('text');
        expect(block).toHaveProperty('confidence');
        expect(block).toHaveProperty('bounding_box');
        expect(block).toHaveProperty('lines');
      }
    } else {
      expect([404, 503]).toContain(resp.status());
    }
  });

  test('full_text is a single string', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/ocr`, {
      multipart: {
        file: { name: 'test.png', mimeType: 'image/png', buffer: PNG_BUFFER },
      },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      expect(typeof body.full_text).toBe('string');
    }
  });

  test('bounding_box has x, y, width, height float fields', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/ocr`, {
      multipart: {
        file: { name: 'test.png', mimeType: 'image/png', buffer: PNG_BUFFER },
      },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      if (body.pages[0]?.blocks?.length > 0) {
        const bb = body.pages[0].blocks[0].bounding_box;
        expect(typeof bb.x).toBe('number');
        expect(typeof bb.y).toBe('number');
        expect(typeof bb.width).toBe('number');
        expect(typeof bb.height).toBe('number');
      }
    }
  });

  test('language parameter accepted (not 422)', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/ocr`, {
      multipart: {
        file: { name: 'test.png', mimeType: 'image/png', buffer: PNG_BUFFER },
        language: 'en',
      },
    });
    expect(resp.status()).not.toBe(422);
  });

  test('detected_languages array in response', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/ocr`, {
      multipart: {
        file: { name: 'test.png', mimeType: 'image/png', buffer: PNG_BUFFER },
      },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      expect(Array.isArray(body.detected_languages)).toBe(true);
    }
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
    const resp = await request.post(`${BASE}/v1/images/segment`, {
      multipart: {
        file: { name: 'test.png', mimeType: 'image/png', buffer: PNG_BUFFER },
        model: 'default',
      },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      expect(body).toHaveProperty('id');
      expect(body).toHaveProperty('model');
      expect(Array.isArray(body.segments)).toBe(true);
      if (body.segments.length > 0) {
        expect(body.segments[0]).toHaveProperty('id');
        expect(body.segments[0]).toHaveProperty('label');
        expect(body.segments[0]).toHaveProperty('score');
      }
    } else {
      expect([404, 503]).toContain(resp.status());
    }
  });

  test('mask_format none returns null mask', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/segment`, {
      multipart: {
        file: { name: 'test.png', mimeType: 'image/png', buffer: PNG_BUFFER },
        model: 'default',
        mask_format: 'none',
      },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      if (body.segments.length > 0) {
        expect(body.segments[0].mask ?? null).toBeNull();
      }
    }
  });

  test('mask_format rle returns RLE mask with counts and size', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/segment`, {
      multipart: {
        file: { name: 'test.png', mimeType: 'image/png', buffer: PNG_BUFFER },
        model: 'default',
        mask_format: 'rle',
      },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      if (body.segments.length > 0 && body.segments[0].mask) {
        expect(body.segments[0].mask.format).toBe('rle');
        expect(body.segments[0].mask).toHaveProperty('counts');
        expect(body.segments[0].mask).toHaveProperty('size');
      }
    }
  });

  test('mask_format raw returns base64 mask', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/segment`, {
      multipart: {
        file: { name: 'test.png', mimeType: 'image/png', buffer: PNG_BUFFER },
        model: 'default',
        mask_format: 'raw',
      },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      if (body.segments.length > 0 && body.segments[0].mask) {
        expect(body.segments[0].mask.format).toBe('raw');
        expect(typeof body.segments[0].mask.counts).toBe('string');
      }
    }
  });

  test('segment scores are between 0.0 and 1.0', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/segment`, {
      multipart: {
        file: { name: 'test.png', mimeType: 'image/png', buffer: PNG_BUFFER },
        model: 'default',
      },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      for (const seg of body.segments) {
        expect(seg.score).toBeGreaterThanOrEqual(0.0);
        expect(seg.score).toBeLessThanOrEqual(1.0);
      }
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

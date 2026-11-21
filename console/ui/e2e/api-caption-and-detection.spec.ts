import { test, expect } from '@playwright/test';
import { PNG_BUFFER } from './fixtures';
import { BASE, requireCachedModel } from './host-capabilities';

// ================================================================
// POST /v1/images/caption
// ================================================================
test.describe('Image Caption', () => {
  test('response: id, model, choices[0].caption, usage', async ({ request }) => {
    await requireCachedModel(request, 'captioner');
    const resp = await request.post(`${BASE}/v1/images/caption`, {
      multipart: {
        file: { name: 'test.png', mimeType: 'image/png', buffer: PNG_BUFFER },
        model: 'default',
      },
    });
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(body).toHaveProperty('id');
    expect(body).toHaveProperty('model');
    expect(Array.isArray(body.choices)).toBe(true);
    expect(body.choices[0]).toHaveProperty('caption');
    expect(body.choices[0]).toHaveProperty('index');
    expect(body.choices[0].finish_reason).toBe('stop');
    expect(body).toHaveProperty('usage');
  });

  test('missing file returns 400', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/caption`, {
      multipart: { model: 'default' },
    });
    expect(resp.status()).toBe(400);
  });

  test('non-image file returns error or graceful handling (not 500)', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/caption`, {
      multipart: {
        file: { name: 'test.txt', mimeType: 'text/plain', buffer: Buffer.from('not an image') },
        model: 'default',
      },
    });
    // Should not be 500 internal error
    expect(resp.status()).not.toBe(500);
  });

  test('usage includes image_tokens', async ({ request }) => {
    await requireCachedModel(request, 'captioner');
    const resp = await request.post(`${BASE}/v1/images/caption`, {
      multipart: {
        file: { name: 'test.png', mimeType: 'image/png', buffer: PNG_BUFFER },
        model: 'default',
      },
    });
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(body.usage).toHaveProperty('image_tokens');
  });
});

// ================================================================
// POST /v1/images/detect
// ================================================================
test.describe('Object Detection', () => {
  test('response: id, model, objects[].label/confidence/bounding_box', async ({ request }) => {
    await requireCachedModel(request, 'detector');
    const resp = await request.post(`${BASE}/v1/images/detect`, {
      multipart: {
        file: { name: 'test.png', mimeType: 'image/png', buffer: PNG_BUFFER },
        model: 'default',
      },
    });
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(body).toHaveProperty('id');
    expect(body).toHaveProperty('model');
    expect(Array.isArray(body.objects)).toBe(true);
    if (body.objects.length > 0) {
      expect(body.objects[0]).toHaveProperty('label');
      expect(body.objects[0]).toHaveProperty('confidence');
      expect(body.objects[0]).toHaveProperty('bounding_box');
      const bb = body.objects[0].bounding_box;
      expect(bb).toHaveProperty('x');
      expect(bb).toHaveProperty('y');
      expect(bb).toHaveProperty('width');
      expect(bb).toHaveProperty('height');
    }
  });

  test('confidence_threshold parameter accepted', async ({ request }) => {
    await requireCachedModel(request, 'detector');
    const resp = await request.post(`${BASE}/v1/images/detect`, {
      multipart: {
        file: { name: 'test.png', mimeType: 'image/png', buffer: PNG_BUFFER },
        model: 'default',
        confidence_threshold: '0.7',
      },
    });
    expect(resp.status()).toBe(200);
  });

  test('output_format coco returns COCO JSON structure', async ({ request }) => {
    await requireCachedModel(request, 'detector');
    const resp = await request.post(`${BASE}/v1/images/detect`, {
      multipart: {
        file: { name: 'test.png', mimeType: 'image/png', buffer: PNG_BUFFER },
        model: 'default',
        output_format: 'coco',
      },
    });
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    // COCO format has images, annotations, categories arrays
    expect(body).toHaveProperty('images');
    expect(body).toHaveProperty('annotations');
    expect(body).toHaveProperty('categories');
  });

  test('missing file returns 400', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/detect`, {
      multipart: { model: 'default' },
    });
    expect(resp.status()).toBe(400);
  });

  test('image with no objects returns an empty objects array', async ({ request }) => {
    await requireCachedModel(request, 'detector');
    const resp = await request.post(`${BASE}/v1/images/detect`, {
      multipart: {
        file: { name: 'test.png', mimeType: 'image/png', buffer: PNG_BUFFER },
        model: 'default',
      },
    });
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    // a 1x1 pixel has no objects
    expect(body.objects).toEqual([]);
  });
});

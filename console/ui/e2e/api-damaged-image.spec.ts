import { test, expect } from '@playwright/test';
import { DAMAGED_PNG_BUFFER } from './fixtures';

const BASE = process.env.API_BASE_URL || 'http://localhost:5000';

// An upload that cannot be decoded is the caller's mistake. Each image endpoint answers 400 naming the problem, not the opaque 500
// an unexpected failure gets.
for (const endpoint of ['ocr', 'segment', 'caption', 'detect']) {
  test(`/v1/images/${endpoint}: a damaged image returns 400`, async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/images/${endpoint}`, {
      multipart: {
        file: { name: 'damaged.png', mimeType: 'image/png', buffer: DAMAGED_PNG_BUFFER },
      },
    });

    // A host that has not got the model for this domain answers 404 or 503 before it reads the image; say so rather than pass.
    test.skip(resp.status() === 404 || resp.status() === 503, 'no model for this endpoint on this host');

    expect(resp.status()).toBe(400);
    const body = await resp.json();
    expect(body.error.message).toContain('can be decoded');
  });
}

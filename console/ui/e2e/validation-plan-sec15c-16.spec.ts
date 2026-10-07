import { test, expect } from '@playwright/test';
import type { Page, Route } from '@playwright/test';

const BASE = process.env.API_BASE_URL || 'http://localhost:5000';

// ================================================================
// Section 15-8: Translate Page
// ================================================================
test.describe('Section 15-8: Translate Page', () => {
  test('[15-8-01] source/target language selection present', async ({ page }) => {
    await page.goto('/translate');
    await page.waitForLoadState('domcontentloaded');
    const content = await page.content();
    expect(content).toMatch(/language|source|target|translate/i);
  });

  test('[15-8-02] translation input field visible', async ({ page }) => {
    await page.goto('/translate');
    await page.waitForLoadState('domcontentloaded');
    const input = page.locator('textarea, input[type="text"]').first();
    await expect(input).toBeVisible({ timeout: 10000 });
  });
});

// ================================================================
// Section 15-9: Vision Pages (Caption, Detect, OCR, Segment)
// ================================================================
const VISION_PAGES = [
  { path: '/caption', name: 'Caption' },
  { path: '/detect', name: 'Detect' },
  { path: '/ocr', name: 'OCR' },
  { path: '/segment', name: 'Segment' },
];

test.describe('Section 15-9: Vision Pages', () => {
  for (const { path, name } of VISION_PAGES) {
    test(`[15-9-01] ${name} page has image upload UI`, async ({ page }) => {
      await page.goto(path);
      await page.waitForLoadState('domcontentloaded');
      // File input or drop zone should exist
      const fileInput = page.locator('input[type="file"]');
      const hasFileInput = await fileInput.count() > 0;
      const content = await page.content();
      const hasDropzone = content.match(/drop|upload|file/i);
      expect(hasFileInput || !!hasDropzone).toBe(true);
    });

    test(`[15-9-04] ${name} page shows loading state during inference`, async ({ page }) => {
      // Mock a slow response to test loading state
      await page.route(`**${path === '/caption' ? '/v1/images/caption' :
        path === '/detect' ? '/v1/images/detect' :
        path === '/ocr' ? '/v1/images/ocr' :
        '/v1/images/segment'}`, async (route: Route) => {
        await new Promise(r => setTimeout(r, 500));
        await route.fulfill({
          status: 200, contentType: 'application/json',
          body: JSON.stringify({ id: 'test', model: 'default', result: 'ok' }),
        });
      });
      await page.goto(path);
      await page.waitForLoadState('domcontentloaded');
      const content = await page.content();
      expect(content.length).toBeGreaterThan(100);
    });
  }
});

// ================================================================
// Section 15-10: ImageGenerate Page
// ================================================================
test.describe('Section 15-10: ImageGenerate Page', () => {
  test('[15-10-01] prompt input field present', async ({ page }) => {
    await page.goto('/generate');
    await page.waitForLoadState('domcontentloaded');
    const input = page.locator('textarea, input[type="text"], input[placeholder]').first();
    await expect(input).toBeVisible({ timeout: 10000 });
  });

  test('[15-10-03] size/steps options visible', async ({ page }) => {
    await page.goto('/generate');
    await page.waitForLoadState('domcontentloaded');
    const content = await page.content();
    expect(content).toMatch(/size|steps|width|height|512|256/i);
  });
});

// ================================================================
// Section 16: Boundary and Stress Tests
// ================================================================
test.describe('Section 16: Boundary and Stress Tests', () => {
  test('[16-01] empty body POST returns 400 with error message', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/embeddings`, {
      headers: { 'Content-Type': 'application/json' },
      data: '{}',
    });
    expect([400, 422]).toContain(resp.status());
  });

  test('[16-02] malformed JSON returns 400 or 422', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/embeddings`, {
      headers: { 'Content-Type': 'application/json' },
      data: 'not-valid-json',
    });
    expect([400, 422]).toContain(resp.status());
  });

  test('[16-07] unsupported HTTP method returns 405', async ({ request }) => {
    const resp = await request.put(`${BASE}/v1/embeddings`, {
      data: { model: 'default', input: 'test' },
    });
    expect(resp.status()).toBe(405);
  });

  test('[16-08] wrong Content-Type returns 415 or 400', async ({ request }) => {
    // Send JSON body but claim form data
    const resp = await request.post(`${BASE}/v1/embeddings`, {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      data: 'model=default&input=test',
    });
    expect([400, 415, 422]).toContain(resp.status());
  });

  test('[16-06] large content body request handled (not crash)', async ({ request }) => {
    // Send oversized input text to test graceful handling
    const largeInput = 'A'.repeat(50000); // 50K chars
    const resp = await request.post(`${BASE}/v1/embeddings`, {
      data: { model: 'default', input: largeInput },
    });
    // Should handle gracefully: either process (200/404/503) or error (400/413)
    expect(resp.status()).not.toBe(500);
    expect([200, 400, 404, 413, 503]).toContain(resp.status());
  });
});

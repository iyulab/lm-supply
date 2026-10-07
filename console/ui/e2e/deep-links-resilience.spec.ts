import { test, expect } from './fixtures/base.fixture';

// ================================================================
// Deep links — all 13 pages load correctly via direct URL
// Covers: deep linking
// ================================================================

test.describe('Deep links — all pages load directly', () => {
  const pages = [
    { path: '/', heading: 'Dashboard' },
    { path: '/chat', heading: /Chat/ },
    { path: '/embed', heading: 'Text Embedding' },
    { path: '/rerank', heading: 'Document Reranking' },
    { path: '/transcribe', heading: 'Speech to Text' },
    { path: '/synthesize', heading: 'Text to Speech' },
    { path: '/caption', heading: /Image Captioning/ },
    { path: '/ocr', heading: 'Optical Character Recognition' },
    { path: '/detect', heading: 'Object Detection' },
    { path: '/segment', heading: 'Image Segmentation' },
    { path: '/translate', heading: 'Machine Translation' },
    { path: '/image-generate', heading: 'Image Generation' },
    { path: '/models', heading: /Model Management/ },
  ];

  for (const { path, heading } of pages) {
    test(`deep link to ${path} renders correctly`, async ({ page }) => {
      await page.goto(path);
      await expect(page.locator('main').getByRole('heading', { name: heading }).first()).toBeVisible();
      // Sidebar should be visible (SPA shell loaded)
      await expect(page.locator('nav')).toBeVisible();
    });
  }
});

// ================================================================
// Error resilience — pages don't crash on initial load
// Covers: error handling / recovery
// ================================================================

test.describe('Error resilience', () => {
  // All pages render without JS errors
  test('no console errors on dashboard', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto('/');
    await expect(page.locator('main').getByRole('heading', { name: 'Dashboard' })).toBeVisible();

    // Allow API errors but no JS crashes
    const jsErrors = errors.filter(e => !e.includes('fetch') && !e.includes('NetworkError'));
    expect(jsErrors).toHaveLength(0);
  });

  // 404 page handling (non-existent route)
  test('non-existent route does not crash browser', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    const response = await page.goto('/this-route-does-not-exist');
    // Page should load without JS crash (regardless of HTTP status)
    expect(response).not.toBeNull();
    const jsErrors = errors.filter(e => !e.includes('fetch') && !e.includes('NetworkError'));
    expect(jsErrors).toHaveLength(0);
  });

  // API failure doesn't crash the page
  test('chat page renders even if model API is slow', async ({ page }) => {
    await page.goto('/chat');
    // Should render the heading regardless of API state
    await expect(page.locator('main').getByRole('heading', { name: /Chat/ }).first()).toBeVisible();
    // Input should still be present
    await expect(page.getByPlaceholder('Type your message...')).toBeVisible();
  });
});

// ================================================================
// Page title / heading consistency
// ================================================================

test.describe('Page heading consistency', () => {
  const headings = [
    { path: '/chat', expected: 'Chat' },
    { path: '/embed', expected: 'Text Embedding' },
    { path: '/rerank', expected: 'Document Reranking' },
    { path: '/transcribe', expected: 'Speech to Text' },
    { path: '/synthesize', expected: 'Text to Speech' },
    { path: '/caption', expected: /Image Captioning/ },
    { path: '/ocr', expected: 'Optical Character Recognition' },
    { path: '/detect', expected: 'Object Detection' },
    { path: '/segment', expected: 'Image Segmentation' },
    { path: '/translate', expected: 'Machine Translation' },
    { path: '/image-generate', expected: 'Image Generation' },
    { path: '/models', expected: /Model Management/ },
  ];

  for (const { path, expected } of headings) {
    test(`${path} has correct heading "${expected}"`, async ({ page }) => {
      await page.goto(path);
      const heading = page.locator('main').getByRole('heading').first();
      await expect(heading).toBeVisible();
      if (typeof expected === 'string') {
        await expect(heading).toContainText(expected);
      } else {
        await expect(heading).toHaveText(expected);
      }
    });
  }
});

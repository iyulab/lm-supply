import { test, expect } from './fixtures/base.fixture';

test.describe('Navigation & Layout', () => {
  // All nav links render
  test('sidebar renders all navigation links', async ({ page }) => {
    await page.goto('/');

    const nav = page.locator('nav');
    await expect(nav).toBeVisible();

    const expectedLinks = [
      'Dashboard', 'Chat', 'Embed', 'Rerank',
      'Transcribe', 'Synthesize', 'Caption', 'OCR',
      'Detect', 'Segment', 'Translate', 'Image Gen',
      'Models',
    ];

    for (const label of expectedLinks) {
      await expect(nav.getByRole('link', { name: label })).toBeVisible();
    }

    // API Docs link (external)
    await expect(nav.getByRole('link', { name: /API Docs/ })).toBeVisible();
  });

  // Navigation works
  test('clicking nav items navigates to correct pages', async ({ page }) => {
    await page.goto('/');

    // Main content area (excludes sidebar)
    const main = page.locator('main');

    const routes: Array<{ label: string; path: string; heading: RegExp }> = [
      { label: 'Chat', path: '/chat', heading: /Chat/ },
      { label: 'Embed', path: '/embed', heading: /Text Embedding/ },
      { label: 'Rerank', path: '/rerank', heading: /Document Reranking/ },
      { label: 'Transcribe', path: '/transcribe', heading: /Speech to Text/ },
      { label: 'Synthesize', path: '/synthesize', heading: /Text to Speech/ },
      { label: 'Caption', path: '/caption', heading: /Image Captioning/ },
      { label: 'OCR', path: '/ocr', heading: /Optical Character Recognition/ },
      { label: 'Detect', path: '/detect', heading: /Object Detection/ },
      { label: 'Segment', path: '/segment', heading: /Image Segmentation/ },
      { label: 'Translate', path: '/translate', heading: /Machine Translation/ },
      { label: 'Image Gen', path: '/image-generate', heading: /Image Generat/ },
      { label: 'Models', path: '/models', heading: /Model Management/ },
    ];

    for (const route of routes) {
      await page.getByRole('link', { name: route.label }).click();
      await expect(page).toHaveURL(route.path);
      await expect(main.locator('h1').first()).toContainText(route.heading);
    }
  });

  // API Docs link opens in new tab
  test('API Docs link has target=_blank', async ({ page }) => {
    await page.goto('/');
    const apiDocsLink = page.getByRole('link', { name: /API Docs/ });
    await expect(apiDocsLink).toHaveAttribute('target', '_blank');
    await expect(apiDocsLink).toHaveAttribute('href', '/swagger');
  });

  // SPA routing (refresh preserves page)
  test('SPA fallback works on page refresh', async ({ page }) => {
    await page.goto('/chat');
    await expect(page).toHaveURL('/chat');

    // Refresh the page
    await page.reload();
    await expect(page).toHaveURL('/chat');
    // Page should still render Chat heading
    await expect(page.getByRole('heading').first()).toBeVisible();
  });

  // Deep link
  test('deep link to /models renders correctly', async ({ page }) => {
    await page.goto('/models');
    await expect(page).toHaveURL('/models');
    await expect(page.getByRole('heading', { name: /Model Management/ })).toBeVisible();
  });

  test('deep link to /embed renders correctly', async ({ page }) => {
    await page.goto('/embed');
    await expect(page).toHaveURL('/embed');
    await expect(page.getByRole('heading', { name: /Embedding/ })).toBeVisible();
  });

  // Version display
  test('version is displayed in sidebar footer', async ({ page }) => {
    await page.goto('/');
    // Version format: v{major}.{minor}.{patch}
    await expect(page.getByText(/^v\d+\.\d+\.\d+$/)).toBeVisible({ timeout: 10_000 });
  });

  // Active link highlighting
  test('active nav link is visually highlighted', async ({ page }) => {
    await page.goto('/embed');
    const embedLink = page.getByRole('link', { name: 'Embed' });
    // Active link should have the primary background class
    await expect(embedLink).toHaveClass(/bg-primary/);

    // Other links should NOT have the primary background
    const chatLink = page.getByRole('link', { name: 'Chat' });
    await expect(chatLink).not.toHaveClass(/bg-primary/);
  });
});

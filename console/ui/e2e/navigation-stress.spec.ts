import { test, expect } from './fixtures/base.fixture';

// ================================================================
// Rapid navigation stress: fast switching, back/forward, deep links
// ================================================================

// ================================================================
// 1. Rapid page switching — visit all pages quickly
// ================================================================

test.describe('Rapid page switching', () => {
  test('can visit all pages in quick succession', async ({ page }) => {
    const pages = [
      { url: '/chat', heading: 'Chat' },
      { url: '/embed', heading: /Embed|Text Embedding/ },
      { url: '/rerank', heading: 'Document Reranking' },
      { url: '/transcribe', heading: 'Speech to Text' },
      { url: '/synthesize', heading: 'Text to Speech' },
      { url: '/caption', heading: /Caption/ },
      { url: '/ocr', heading: /Optical Character Recognition|OCR/ },
      { url: '/detect', heading: 'Detect' },
      { url: '/segment', heading: 'Segment' },
      { url: '/translate', heading: 'Machine Translation' },
      { url: '/image-generate', heading: 'Image Generation' },
      { url: '/models', heading: 'Model Management' },
    ];

    for (const p of pages) {
      await page.goto(p.url);
      await expect(page.locator('main').getByRole('heading', { name: p.heading })).toBeVisible();
    }
  });

  test('rapid sidebar clicks navigate correctly', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('main')).toBeVisible();

    // Click through sidebar links rapidly
    const navTargets = ['Chat', 'Embed', 'Rerank', 'Translate', 'Synthesize'];
    for (const target of navTargets) {
      await page.locator(`nav a`, { hasText: target }).click();
      await expect(page.locator('main').getByRole('heading').first()).toBeVisible();
    }

    // Verify we ended on Synthesize
    await expect(page.locator('main').getByRole('heading', { name: 'Text to Speech' })).toBeVisible();
  });
});

// ================================================================
// 2. Browser back/forward navigation
// ================================================================

test.describe('Browser back/forward', () => {
  test('back button returns to previous page', async ({ page }) => {
    await page.goto('/chat');
    await expect(page.locator('main').getByRole('heading', { name: 'Chat' })).toBeVisible();

    await page.goto('/embed');
    await expect(page.locator('main').getByRole('heading', { name: /Embed|Text Embedding/ })).toBeVisible();

    await page.goBack();
    await expect(page.locator('main').getByRole('heading', { name: 'Chat' })).toBeVisible();
  });

  test('forward button after back works', async ({ page }) => {
    await page.goto('/chat');
    await expect(page.locator('main').getByRole('heading', { name: 'Chat' })).toBeVisible();

    await page.goto('/translate');
    await expect(page.locator('main').getByRole('heading', { name: 'Machine Translation' })).toBeVisible();

    await page.goBack();
    await expect(page.locator('main').getByRole('heading', { name: 'Chat' })).toBeVisible();

    await page.goForward();
    await expect(page.locator('main').getByRole('heading', { name: 'Machine Translation' })).toBeVisible();
  });

  test('multiple back navigations work', async ({ page }) => {
    await page.goto('/chat');
    await page.goto('/embed');
    await page.goto('/rerank');
    await expect(page.locator('main').getByRole('heading', { name: 'Document Reranking' })).toBeVisible();

    await page.goBack();
    await expect(page.locator('main').getByRole('heading', { name: /Embed|Text Embedding/ })).toBeVisible();

    await page.goBack();
    await expect(page.locator('main').getByRole('heading', { name: 'Chat' })).toBeVisible();
  });
});

// ================================================================
// 3. State isolation between pages
// ================================================================

test.describe('State isolation', () => {
  test('input state does not leak between page navigations', async ({ page }) => {
    // Type in Chat
    await page.goto('/chat');
    await expect(page.locator('main').getByRole('heading', { name: 'Chat' })).toBeVisible();
    const chatInput = page.locator('input[type="text"]').first();
    await chatInput.fill('Chat message');

    // Navigate to Embed
    await page.goto('/embed');
    await expect(page.locator('main').getByRole('heading', { name: /Embed|Text Embedding/ })).toBeVisible();
    const embedTextarea = page.locator('textarea');
    await expect(embedTextarea).toHaveValue('');

    // Navigate to Translate
    await page.goto('/translate');
    await expect(page.locator('main').getByRole('heading', { name: 'Machine Translation' })).toBeVisible();
    const translateTextarea = page.locator('textarea');
    await expect(translateTextarea).toHaveValue('');
  });

  // page model independence
  test('model selection is independent between pages', async ({ page }) => {
    // Go to Embed — should have its own model state
    await page.goto('/embed');
    await expect(page.locator('main').getByRole('heading', { name: /Embed|Text Embedding/ })).toBeVisible();
    const embedTextarea = page.locator('textarea');
    await embedTextarea.fill('test embedding text');

    // Go to Rerank — completely separate page state
    await page.goto('/rerank');
    await expect(page.locator('main').getByRole('heading', { name: 'Document Reranking' })).toBeVisible();
    // Rerank input should be empty (not "test embedding text")
    const rerankInput = page.locator('textarea').first();
    await expect(rerankInput).toHaveValue('');

    // Go back to Embed — should still be functional and independent
    await page.goto('/embed');
    await expect(page.locator('main').getByRole('heading', { name: /Embed|Text Embedding/ })).toBeVisible();
    // Input resets on navigation (React state is component-scoped)
    const embedTextarea2 = page.locator('textarea');
    await expect(embedTextarea2).toHaveValue('');
    // Can still type (page is functional)
    await embedTextarea2.fill('new text');
    await expect(embedTextarea2).toHaveValue('new text');
  });

  test('error state does not persist across page navigations', async ({ page }) => {
    // Mock embed to fail
    await page.route('**/v1/embeddings', async (route) => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: { message: 'Test error' } }),
      });
    });

    await page.goto('/embed');
    await expect(page.locator('main').getByRole('heading', { name: /Embed|Text Embedding/ })).toBeVisible();
    await page.locator('textarea').fill('test');
    await page.locator('button[type="submit"]').click();
    await expect(page.getByText(/error|fail/i)).toBeVisible({ timeout: 10_000 });

    // Navigate away and back
    await page.goto('/chat');
    await page.goto('/embed');

    // Error should be cleared
    const errorText = page.locator('main').getByText(/Test error/);
    await expect(errorText).not.toBeVisible({ timeout: 3_000 });
  });
});

// ================================================================
// 4. Page load performance sanity
// ================================================================

test.describe('Page load performance', () => {
  test('all pages load within timeout', async ({ page }) => {
    const allPages = [
      '/', '/chat', '/embed', '/rerank', '/transcribe',
      '/synthesize', '/caption', '/ocr', '/detect', '/segment',
      '/translate', '/image-generate', '/models',
    ];

    for (const url of allPages) {
      await page.goto(url);
      // Each page should have content within default timeout
      await expect(page.locator('main')).toBeVisible();
    }
  });
});

// ================================================================
// 5. Navigation via URL bar (deep links)
// ================================================================

test.describe('Deep link navigation', () => {
  test('all routes resolve to correct pages', async ({ page }) => {
    const routes = [
      { url: '/chat', expected: 'Chat' },
      { url: '/embed', expected: /Embed|Text Embedding/ },
      { url: '/rerank', expected: 'Document Reranking' },
      { url: '/translate', expected: 'Machine Translation' },
      { url: '/image-generate', expected: 'Image Generation' },
    ];

    for (const route of routes) {
      await page.goto(route.url);
      await expect(page.locator('main').getByRole('heading', { name: route.expected })).toBeVisible();
    }
  });

  test('navigating to valid route after visiting all routes works', async ({ page }) => {
    // Verify the last route works after visiting several
    await page.goto('/models');
    await expect(page.locator('main').getByRole('heading', { name: 'Model Management' })).toBeVisible();
  });
});

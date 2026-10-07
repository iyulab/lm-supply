import { test, expect } from './fixtures/base.fixture';

// ================================================================
// Accessibility: aria-labels on icon-only buttons
// ================================================================

test.describe('Accessibility — icon button labels', () => {
  test('Chat send button has aria-label', async ({ page }) => {
    await page.goto('/chat');
    await expect(page.locator('main').getByRole('heading', { name: /Chat/ }).first()).toBeVisible();

    const sendButton = page.locator('button[type="submit"]');
    await expect(sendButton).toHaveAttribute('aria-label', 'Send message');
  });

  test('Image Generate randomize seed button has aria-label', async ({ page }) => {
    await page.goto('/image-generate');
    await expect(page.locator('main').getByRole('heading', { name: 'Image Generation' })).toBeVisible();

    // Open advanced settings
    await page.getByRole('button', { name: /Advanced Settings/i }).click();

    const randomizeButton = page.getByRole('button', { name: 'Randomize seed' });
    await expect(randomizeButton).toBeVisible();
    await expect(randomizeButton).toHaveAttribute('aria-label', 'Randomize seed');
  });
});

// ================================================================
// Accessibility: form label associations
// ================================================================

test.describe('Accessibility — form labels', () => {
  test('Translate page has labeled form elements', async ({ page }) => {
    await page.goto('/translate');
    await expect(page.locator('main').getByRole('heading', { name: 'Machine Translation' })).toBeVisible();

    // Model label exists
    await expect(page.locator('label', { hasText: 'Model' })).toBeVisible();
    // Source Text label exists
    await expect(page.locator('label', { hasText: 'Source Text' })).toBeVisible();
  });

  test('Embed page has labeled form elements', async ({ page }) => {
    await page.goto('/embed');
    await expect(page.locator('main').getByRole('heading', { name: 'Text Embedding' })).toBeVisible();

    await expect(page.locator('label', { hasText: 'Model' })).toBeVisible();
    await expect(page.locator('label', { hasText: /Texts/ })).toBeVisible();
  });

  test('Rerank page has labeled form elements', async ({ page }) => {
    await page.goto('/rerank');
    await expect(page.locator('main').getByRole('heading', { name: 'Document Reranking' })).toBeVisible();

    await expect(page.locator('label', { hasText: 'Model' })).toBeVisible();
    await expect(page.locator('label', { hasText: 'Query' })).toBeVisible();
    await expect(page.locator('label', { hasText: /Documents/ })).toBeVisible();
    await expect(page.locator('label', { hasText: 'Top K' })).toBeVisible();
  });

  test('Detect page has labeled form elements', async ({ page }) => {
    await page.goto('/detect');
    await expect(page.locator('main').getByRole('heading', { name: 'Object Detection' })).toBeVisible();

    await expect(page.locator('label', { hasText: 'Model ID' })).toBeVisible();
    await expect(page.locator('label', { hasText: /Confidence Threshold/ })).toBeVisible();
  });

  test('Image Generate page has labeled form elements', async ({ page }) => {
    await page.goto('/image-generate');
    await expect(page.locator('main').getByRole('heading', { name: 'Image Generation' })).toBeVisible();

    await expect(page.locator('label', { hasText: 'Prompt' })).toBeVisible();
    await expect(page.locator('label', { hasText: 'Model' })).toBeVisible();
    await expect(page.locator('label', { hasText: 'Size' })).toBeVisible();
  });
});

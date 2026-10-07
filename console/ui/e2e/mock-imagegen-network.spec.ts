import { test, expect } from './fixtures/base.fixture';
import { mockJsonEndpoint, mockImageGenerationResponse, mockJsonError } from './fixtures/api-mocks';

// ================================================================
// ImageGen — download button behavior
// Covers: download button creates PNG file
// ================================================================

test.describe('ImageGen — download flow', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/image-generate');
    await expect(page.locator('main').getByRole('heading', { name: 'Image Generation' })).toBeVisible();
  });

  // Download PNG button creates download link
  test('download button creates a PNG download link', async ({ page }) => {
    const mockResponse = mockImageGenerationResponse();
    await mockJsonEndpoint(page, '**/v1/images/generate', mockResponse);

    await page.locator('textarea').fill('A beautiful landscape');
    await page.getByRole('button', { name: /Generate Image/ }).click();

    // Wait for result to appear
    await expect(page.getByText('Generation Details')).toBeVisible({ timeout: 15_000 });

    // Download button should be visible
    const downloadBtn = page.getByRole('button', { name: /Download/ });
    await expect(downloadBtn).toBeVisible();

    // Set up download listener and click
    const [download] = await Promise.all([
      page.waitForEvent('download'),
      downloadBtn.click(),
    ]);

    // Verify the download file name matches the pattern
    expect(download.suggestedFilename()).toMatch(/generated-.*\.png/);
  });
});

// ================================================================
// Network error handling and recovery
// Covers: network error shows message, error recovery
// ================================================================

test.describe('Network error handling', () => {
  // API error shows error message (not blank page)
  test('embed page shows error message on API failure', async ({ page }) => {
    await page.goto('/embed');
    await expect(page.locator('main').getByRole('heading', { name: /Text Embedding/ })).toBeVisible();
    await expect(page.getByText('Loading models...')).toBeHidden({ timeout: 15_000 });

    // Mock embed endpoint to return 500
    await mockJsonError(page, '**/v1/embeddings', 'Internal server error: model unavailable');

    // Type text and submit
    await page.locator('textarea').fill('Test text');
    await page.getByRole('button', { name: /Generate Embeddings/ }).click();

    // Error message should appear — page should NOT crash
    await expect(page.getByText(/Internal server error/)).toBeVisible({ timeout: 10_000 });

    // Page should still be functional (heading still visible)
    await expect(page.locator('main').getByRole('heading', { name: /Text Embedding/ })).toBeVisible();
  });

  // Chat shows error as assistant message
  test('chat page shows error as assistant message on API failure', async ({ page }) => {
    await page.goto('/chat');
    await expect(page.locator('main').getByRole('heading', { name: /Chat/ }).first()).toBeVisible();
    await expect(page.getByText('Loading models...')).toBeHidden({ timeout: 15_000 });

    // Mock chat endpoint to return error
    await page.route('**/v1/chat/completions', async (route) => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: { message: 'Model failed to load' } }),
      });
    });

    await page.getByPlaceholder('Type your message...').fill('Hello');
    await page.locator('button[type="submit"]').click();

    // Error should appear as message (not crash)
    await expect(page.getByText(/Model failed to load|Error|error/)).toBeVisible({ timeout: 10_000 });
  });

  // Error recovery: operations resume after error
  test('operations resume normally after error recovery', async ({ page }) => {
    await page.goto('/embed');
    await expect(page.locator('main').getByRole('heading', { name: /Text Embedding/ })).toBeVisible();
    await expect(page.getByText('Loading models...')).toBeHidden({ timeout: 15_000 });

    // First: mock embed to fail
    await mockJsonError(page, '**/v1/embeddings', 'Service temporarily unavailable');

    await page.locator('textarea').fill('First attempt');
    await page.getByRole('button', { name: /Generate Embeddings/ }).click();

    // Error should appear
    await expect(page.getByText(/Service temporarily unavailable/)).toBeVisible({ timeout: 10_000 });

    // Now: unmock and re-mock with success
    await page.unroute('**/v1/embeddings');
    const successResponse = {
      object: 'list',
      data: [{ object: 'embedding', index: 0, embedding: [0.1, 0.2, 0.3, 0.4, 0.5] }],
      model: 'mock-embedder',
      usage: { prompt_tokens: 3, total_tokens: 3 },
    };
    await mockJsonEndpoint(page, '**/v1/embeddings', successResponse);

    // Retry
    await page.locator('textarea').fill('Second attempt');
    await page.getByRole('button', { name: /Generate Embeddings/ }).click();

    // Should succeed this time
    await expect(page.getByText(/Results \(1 embedding/)).toBeVisible({ timeout: 10_000 });
  });
});

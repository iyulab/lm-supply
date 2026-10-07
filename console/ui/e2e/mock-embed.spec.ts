import { test, expect } from './fixtures/base.fixture';
import { mockJsonEndpoint, mockEmbedResponse, mockJsonError } from './fixtures/api-mocks';

// ================================================================
// Embed page — mock API tests for result display
// Covers: embedding generation, vector display, elapsed time
// ================================================================

test.describe('Embed — mock inference results', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/embed');
    await expect(page.locator('main').getByRole('heading', { name: 'Text Embedding' })).toBeVisible();
    // Wait for model selector to load
    await expect(page.getByText('Loading models...')).toBeHidden({ timeout: 15_000 });
  });

  // [4-17] Single text embedding shows count and dimensions
  test('[4-17] embedding results show count and dimensions', async ({ page }) => {
    const texts = ['Hello world', 'Test sentence'];
    const mockResponse = mockEmbedResponse(texts, 384);

    // Mock the embeddings endpoint
    await mockJsonEndpoint(page, '**/v1/embeddings', mockResponse);

    // Fill textarea
    const textarea = page.locator('textarea');
    await textarea.fill(texts.join('\n'));

    // Click submit
    await page.getByRole('button', { name: /Generate Embeddings/ }).click();

    // Verify results heading: "Results (2 embeddings, 384 dimensions)"
    await expect(page.getByText(/Results.*2 embeddings.*384 dimensions/)).toBeVisible({ timeout: 10_000 });
  });

  // [4-19] Vector preview shows first values and dimension count
  test('[4-19] vector preview shows first values and dims', async ({ page }) => {
    const texts = ['Single text'];
    const mockResponse = mockEmbedResponse(texts, 384);

    await mockJsonEndpoint(page, '**/v1/embeddings', mockResponse);

    const textarea = page.locator('textarea');
    await textarea.fill(texts[0]);
    await page.getByRole('button', { name: /Generate Embeddings/ }).click();

    // Should show "Text 1" label
    await expect(page.getByText('Text 1')).toBeVisible({ timeout: 10_000 });

    // Should show vector preview: "[x.xxxx, y.yyyy, ...] (384 dims)"
    await expect(page.getByText(/\[.*\.\d{4}.*\].*384 dims/)).toBeVisible();
  });

  // [4-20] Elapsed time shown after embedding
  test('[4-20] elapsed time is shown after generation', async ({ page }) => {
    const mockResponse = mockEmbedResponse(['Test'], 384);
    await mockJsonEndpoint(page, '**/v1/embeddings', mockResponse);

    await page.locator('textarea').fill('Test text');
    await page.getByRole('button', { name: /Generate Embeddings/ }).click();

    // Should show elapsed time in ms
    await expect(page.getByText(/\d+ ms/)).toBeVisible({ timeout: 10_000 });
  });

  // [4-18] Multiple texts generate multiple embeddings
  test('[4-18] multiple texts generate multiple embedding results', async ({ page }) => {
    const texts = ['First text', 'Second text', 'Third text'];
    const mockResponse = mockEmbedResponse(texts, 256);

    await mockJsonEndpoint(page, '**/v1/embeddings', mockResponse);

    await page.locator('textarea').fill(texts.join('\n'));
    await page.getByRole('button', { name: /Generate Embeddings/ }).click();

    // Should show results for all 3 texts
    await expect(page.getByText(/3 embeddings.*256 dimensions/)).toBeVisible({ timeout: 10_000 });
    await expect(page.getByText('Text 1')).toBeVisible();
    await expect(page.getByText('Text 2')).toBeVisible();
    await expect(page.getByText('Text 3')).toBeVisible();
  });

  // Error display
  test('API error shows error message', async ({ page }) => {
    await mockJsonError(page, '**/v1/embeddings', 'Model failed to load: not found');

    await page.locator('textarea').fill('Some text');
    await page.getByRole('button', { name: /Generate Embeddings/ }).click();

    // Should show error message
    await expect(page.getByText(/Model failed to load/)).toBeVisible({ timeout: 10_000 });
  });
});

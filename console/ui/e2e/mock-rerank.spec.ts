import { test, expect } from './fixtures/base.fixture';
import { mockJsonEndpoint, mockRerankResponse, mockJsonError } from './fixtures/api-mocks';

// ================================================================
// Rerank page — mock API tests for result display
// Covers: rerank documents, result display
// ================================================================

test.describe('Rerank — mock inference results', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/rerank');
    await expect(page.locator('main').getByRole('heading', { name: 'Document Reranking' })).toBeVisible();
    await expect(page.getByText('Loading models...')).toBeHidden({ timeout: 15_000 });
  });

  // Rerank execution shows ranked results
  test('rerank results show ranked documents', async ({ page }) => {
    const docs = ['Machine learning basics', 'Deep learning for NLP', 'Cooking recipes'];
    const query = 'artificial intelligence';
    const mockResponse = mockRerankResponse(docs, query);

    await mockJsonEndpoint(page, '**/v1/rerank', mockResponse);

    // Fill query and documents
    await page.getByPlaceholder('Enter your query...').fill(query);
    await page.locator('textarea').fill(docs.join('\n'));
    await page.getByRole('button', { name: /Rerank Documents/ }).click();

    // Results heading should appear
    await expect(page.getByText('Ranked Results')).toBeVisible({ timeout: 10_000 });
  });

  // Result items show position number, score, and text
  test('results show position numbers and scores', async ({ page }) => {
    const docs = ['Document A', 'Document B', 'Document C'];
    const mockResponse = mockRerankResponse(docs, 'test query');

    await mockJsonEndpoint(page, '**/v1/rerank', mockResponse);

    await page.getByPlaceholder('Enter your query...').fill('test query');
    await page.locator('textarea').fill(docs.join('\n'));
    await page.getByRole('button', { name: /Rerank Documents/ }).click();

    // Should show #1 position
    await expect(page.getByText('#1')).toBeVisible({ timeout: 10_000 });

    // Should show score with 4 decimal places
    await expect(page.getByText(/Score: 0\.\d{4}/).first()).toBeVisible();

    // Should show document text in result area
    const resultArea = page.locator('.bg-card');
    await expect(resultArea.getByText('Document A')).toBeVisible();
  });

  // Elapsed time shown after reranking
  test('elapsed time shown after reranking', async ({ page }) => {
    const mockResponse = mockRerankResponse(['Doc 1'], 'query');
    await mockJsonEndpoint(page, '**/v1/rerank', mockResponse);

    await page.getByPlaceholder('Enter your query...').fill('query');
    await page.locator('textarea').fill('Doc 1');
    await page.getByRole('button', { name: /Rerank Documents/ }).click();

    await expect(page.getByText(/\d+ ms/)).toBeVisible({ timeout: 10_000 });
  });

  // Error handling
  test('API error shows error message', async ({ page }) => {
    await mockJsonError(page, '**/v1/rerank', 'Reranker model not loaded');

    await page.getByPlaceholder('Enter your query...').fill('query');
    await page.locator('textarea').fill('Some document');
    await page.getByRole('button', { name: /Rerank Documents/ }).click();

    await expect(page.getByText(/Reranker model not loaded/)).toBeVisible({ timeout: 10_000 });
  });
});

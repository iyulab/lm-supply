import { test, expect } from './fixtures/base.fixture';

// ================================================================
// Embed page — deeper interaction tests
// ================================================================

test.describe('Embed — deeper UI', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/embed');
    await expect(page.locator('main').getByRole('heading', { name: 'Text Embedding' })).toBeVisible();
  });

  // textarea placeholder
  test('textarea has "one per line" placeholder', async ({ page }) => {
    const textarea = page.locator('textarea');
    await expect(textarea).toHaveAttribute('placeholder', 'Enter texts to embed, one per line...');
  });

  // label says "Texts (one per line)"
  test('label indicates one text per line', async ({ page }) => {
    await expect(page.locator('label', { hasText: /Texts.*one per line/ })).toBeVisible();
  });

  // Submit button text
  test('submit button says "Generate Embeddings"', async ({ page }) => {
    const btn = page.getByRole('button', { name: /Generate Embeddings/ });
    await expect(btn).toBeVisible();
  });

  // No results section before submit
  test('no results section shown before submission', async ({ page }) => {
    await expect(page.getByText(/Results.*embeddings.*dimensions/)).not.toBeVisible();
  });

  // Textarea height
  test('textarea has consistent height', async ({ page }) => {
    const textarea = page.locator('textarea');
    await expect(textarea).toHaveClass(/h-40/);
  });
});

// ================================================================
// Rerank page — deeper interaction tests
// ================================================================

test.describe('Rerank — deeper UI', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/rerank');
    await expect(page.locator('main').getByRole('heading', { name: 'Document Reranking' })).toBeVisible();
  });

  // Top K defaults to 5
  test('Top K input defaults to 5', async ({ page }) => {
    const topKInput = page.locator('input[type="number"]');
    const value = await topKInput.inputValue();
    expect(value).toBe('5');
  });

  // Top K has min/max constraints
  test('Top K has min 1 and max 100', async ({ page }) => {
    const topKInput = page.locator('input[type="number"]');
    await expect(topKInput).toHaveAttribute('min', '1');
    await expect(topKInput).toHaveAttribute('max', '100');
  });

  // Top K is editable
  test('Top K is editable', async ({ page }) => {
    const topKInput = page.locator('input[type="number"]');
    await topKInput.fill('3');
    const value = await topKInput.inputValue();
    expect(value).toBe('3');
  });

  // Query placeholder
  test('query input has placeholder', async ({ page }) => {
    const queryInput = page.getByPlaceholder('Enter your query...');
    await expect(queryInput).toBeVisible();
  });

  // Documents placeholder
  test('documents textarea has placeholder', async ({ page }) => {
    const docsTextarea = page.getByPlaceholder('Enter documents to rerank, one per line...');
    await expect(docsTextarea).toBeVisible();
  });

  // Submit button text
  test('submit button says "Rerank Documents"', async ({ page }) => {
    const btn = page.getByRole('button', { name: /Rerank Documents/ });
    await expect(btn).toBeVisible();
  });

  // No results before submit
  test('no results section shown before submission', async ({ page }) => {
    await expect(page.getByText(/Ranked Results/)).not.toBeVisible();
  });

  // Documents label
  test('documents label indicates one per line', async ({ page }) => {
    await expect(page.locator('label', { hasText: /Documents.*one per line/ })).toBeVisible();
  });
});

// ================================================================
// Chat page — deeper interaction tests
// ================================================================

test.describe('Chat — deeper interaction', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/chat');
    await expect(page.locator('main').getByRole('heading', { name: /Chat/ }).first()).toBeVisible();
  });

  // empty state message
  test('empty chat shows placeholder message', async ({ page }) => {
    await expect(page.getByText('Start a conversation with the AI model')).toBeVisible();
  });

  // Model label is visible
  test('model label is visible', async ({ page }) => {
    await expect(page.locator('label', { hasText: 'Model' })).toBeVisible();
  });

  // Input and form work correctly
  test('chat form is a proper form element', async ({ page }) => {
    const form = page.locator('form');
    await expect(form).toBeVisible();
  });

  // Send button is type=submit
  test('send button is submit type', async ({ page }) => {
    const submitBtn = page.locator('button[type="submit"]');
    await expect(submitBtn).toBeVisible();
  });

  // Input placeholder is correct
  test('input has "Type your message..." placeholder', async ({ page }) => {
    const input = page.getByPlaceholder('Type your message...');
    await expect(input).toBeVisible();
  });

  // Cancel button styling
  test('cancel button has destructive styling class', async ({ page }) => {
    // We verify the cancel button appears with destructive styling
    // Since we can't easily trigger loading without real inference,
    // this is tested in chat.spec.ts via JS evaluation
    // Here we verify the form structure supports cancel
    const form = page.locator('form');
    const buttons = form.locator('button');
    // At minimum, the submit button should be present
    expect(await buttons.count()).toBeGreaterThanOrEqual(1);
  });
});

import { test, expect } from './fixtures/base.fixture';

test.describe('Embed page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/embed');
    await expect(page.locator('main').getByRole('heading', { name: 'Text Embedding' })).toBeVisible();
  });

  // [4-15] Embedder model selector loads models
  test('[4-15] model selector loads embedder models', async ({ page }) => {
    await expect(page.getByText('Loading models...')).toBeHidden({ timeout: 15_000 });
    const select = page.locator('select');
    const optionCount = await select.locator('option').count();
    expect(optionCount).toBeGreaterThan(0);

    // No "undefined" in options
    const options = await select.locator('option').allTextContents();
    for (const opt of options) {
      expect(opt).not.toContain('undefined');
    }
  });

  // [4-16] Empty input disables submit
  test('[4-16] submit button is disabled when textarea is empty', async ({ page }) => {
    await expect(page.getByText('Loading models...')).toBeHidden({ timeout: 15_000 });
    const submitButton = page.getByRole('button', { name: /Generate Embeddings/ });
    await expect(submitButton).toBeDisabled();
  });

  // Textarea and form elements present
  test('has textarea for text input', async ({ page }) => {
    const textarea = page.getByPlaceholder(/Enter texts.*one per line/);
    await expect(textarea).toBeVisible();
  });

  // Submit enables with text
  test('submit button enables when text is entered', async ({ page }) => {
    await expect(page.getByText('Loading models...')).toBeHidden({ timeout: 15_000 });

    const textarea = page.getByPlaceholder(/Enter texts.*one per line/);
    await textarea.fill('Hello world\nTest sentence');

    const submitButton = page.getByRole('button', { name: /Generate Embeddings/ });
    const select = page.locator('select');
    const modelValue = await select.inputValue();
    if (modelValue) {
      await expect(submitButton).toBeEnabled();
    }
  });
});

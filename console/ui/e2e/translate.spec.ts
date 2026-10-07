import { test, expect } from './fixtures/base.fixture';

test.describe('Translate page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/translate');
    await expect(page.locator('main').getByRole('heading', { name: 'Machine Translation' })).toBeVisible();
  });

  // Language pair selector loads translation models
  test('model selector is populated with language pairs', async ({ page }) => {
    const select = page.locator('select');
    await expect(select).toBeVisible();

    // Wait for options to be populated from API
    await expect(select.locator('option')).not.toHaveCount(0, { timeout: 5_000 });

    const options = await select.locator('option').allTextContents();
    expect(options.length).toBeGreaterThan(0);

    // No "undefined" in options
    for (const opt of options) {
      expect(opt).not.toContain('undefined');
    }
  });

  // Source/target language direction labels shown
  test('shows source and target language labels', async ({ page }) => {
    // Source Text label should exist
    await expect(page.getByText('Source Text')).toBeVisible();
    // Translation label should exist (use exact label locator to avoid strict mode)
    const translationLabel = page.locator('label', { hasText: /^Translation/ });
    await expect(translationLabel).toBeVisible();
  });

  // Translate button disabled when source empty
  test('translate button is disabled when source text is empty', async ({ page }) => {
    const submitButton = page.getByRole('button', { name: /Translate/i });
    await expect(submitButton).toBeDisabled();
  });

  // Source textarea present
  test('has source text textarea with placeholder', async ({ page }) => {
    const textarea = page.getByPlaceholder(/Enter text to translate/);
    await expect(textarea).toBeVisible();
  });

  // Submit enables with text
  test('translate button enables when text is entered', async ({ page }) => {
    const textarea = page.getByPlaceholder(/Enter text to translate/);
    await textarea.fill('Hello, world!');

    const submitButton = page.getByRole('button', { name: /Translate/i });
    await expect(submitButton).toBeEnabled();
  });

  // Translation output placeholder
  test('shows translation placeholder text', async ({ page }) => {
    await expect(page.getByText('Translation will appear here...')).toBeVisible();
  });
});

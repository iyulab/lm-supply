import { test, expect } from './fixtures/base.fixture';

test.describe('Synthesize page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/synthesize');
    await expect(page.locator('main').getByRole('heading', { name: 'Text to Speech' })).toBeVisible();
  });

  // Synthesizer model selector loads Piper models
  test('model selector loads synthesizer models', async ({ page }) => {
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

  // Generate button disabled when text empty
  test('submit button is disabled when textarea is empty', async ({ page }) => {
    await expect(page.getByText('Loading models...')).toBeHidden({ timeout: 15_000 });
    const submitButton = page.getByRole('button', { name: /Generate Speech/ });
    await expect(submitButton).toBeDisabled();
  });

  // Text input present
  test('has textarea for text input', async ({ page }) => {
    const textarea = page.getByPlaceholder(/Enter text to convert to speech/);
    await expect(textarea).toBeVisible();
  });

  // Submit enables with text
  test('submit button enables when text is entered', async ({ page }) => {
    await expect(page.getByText('Loading models...')).toBeHidden({ timeout: 15_000 });

    const textarea = page.getByPlaceholder(/Enter text to convert to speech/);
    await textarea.fill('Hello, this is a test.');

    const submitButton = page.getByRole('button', { name: /Generate Speech/ });
    const select = page.locator('select');
    const modelValue = await select.inputValue();
    if (modelValue) {
      await expect(submitButton).toBeEnabled();
    }
  });
});

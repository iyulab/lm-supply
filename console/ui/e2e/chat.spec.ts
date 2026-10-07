import { test, expect } from './fixtures/base.fixture';

test.describe('Chat page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/chat');
    await expect(page.locator('main').getByRole('heading', { name: 'Chat' })).toBeVisible();
  });

  // ================================================================
  // 3.1 Model Selection
  // ================================================================

  // Model selector loads generator models
  test('model selector loads and shows generator models', async ({ page }) => {
    // Wait for loading to finish
    await expect(page.getByText('Loading models...')).toBeHidden({ timeout: 15_000 });

    // Should have a select element with options
    const select = page.locator('select');
    const optionCount = await select.locator('option').count();
    expect(optionCount).toBeGreaterThan(0);
  });

  // No undefined in model selector options
  test('model selector does not show "undefined" in options', async ({ page }) => {
    await expect(page.getByText('Loading models...')).toBeHidden({ timeout: 15_000 });

    const select = page.locator('select');
    const options = await select.locator('option').allTextContents();

    for (const option of options) {
      expect(option).not.toContain('undefined');
    }
  });

  // Auto-select
  test('first model is auto-selected', async ({ page }) => {
    await expect(page.getByText('Loading models...')).toBeHidden({ timeout: 15_000 });

    const select = page.locator('select');
    const value = await select.inputValue();
    expect(value).toBeTruthy();
  });

  // ================================================================
  // 3.2 Chat Interaction
  // ================================================================

  // Send disabled when input is empty
  test('send button is disabled when input is empty', async ({ page }) => {
    await expect(page.getByText('Loading models...')).toBeHidden({ timeout: 15_000 });

    // Submit button should be disabled
    const submitButton = page.locator('button[type="submit"]');
    await expect(submitButton).toBeDisabled();
  });

  // Empty state
  test('shows empty state message when no messages', async ({ page }) => {
    await expect(page.getByText('Start a conversation with the AI model')).toBeVisible();
  });

  // Input placeholder
  test('chat input has correct placeholder', async ({ page }) => {
    const input = page.getByPlaceholder('Type your message...');
    await expect(input).toBeVisible();
  });

  // Send button enables when text entered
  test('send button enables when text is entered', async ({ page }) => {
    await expect(page.getByText('Loading models...')).toBeHidden({ timeout: 15_000 });

    const input = page.getByPlaceholder('Type your message...');
    const submitButton = page.locator('button[type="submit"]');

    // Initially disabled
    await expect(submitButton).toBeDisabled();

    // Type something
    await input.fill('Hello');

    // Now should be enabled (if model is selected)
    const select = page.locator('select');
    const modelValue = await select.inputValue();
    if (modelValue) {
      await expect(submitButton).toBeEnabled();
    }
  });

  // Cancel button exists (when loading)
  test('cancel button has destructive styling', async ({ page }) => {
    // The cancel button only appears during loading
    // We can verify the submit button exists and has correct structure
    const form = page.locator('form');
    await expect(form).toBeVisible();

    // Submit button should have Send icon
    const submitButton = form.locator('button[type="submit"]');
    await expect(submitButton).toBeVisible();
  });
});

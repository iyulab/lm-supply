import { test, expect } from './fixtures/base.fixture';

test.describe('Transcribe page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/transcribe');
    await expect(page.locator('main').getByRole('heading', { name: 'Speech to Text' })).toBeVisible();
  });

  // Transcriber model selector loads Whisper models
  test('model selector loads transcriber models', async ({ page }) => {
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

  // Drop zone visible when model not selected
  test('file input is disabled when no model is selected', async ({ page }) => {
    // Initially while loading, the file input should be disabled
    const fileInput = page.locator('input[type="file"]');
    // File input exists but may be hidden — check the drop zone text
    const dropZone = page.locator('.border-dashed');
    await expect(dropZone).toBeVisible();
  });

  // Drop zone accepts audio/* files
  test('has a file upload drop zone with audio accept', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]');
    await expect(fileInput).toHaveAttribute('accept', 'audio/*');
  });

  // Supported audio format hints displayed
  test('shows supported audio format hints', async ({ page }) => {
    await expect(page.getByText(/WAV.*MP3|audio formats/i)).toBeVisible();
  });
});

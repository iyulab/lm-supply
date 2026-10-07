import { test, expect } from './fixtures/base.fixture';

// ================================================================
// Synthesize page — deeper interaction tests
// ================================================================

test.describe('Synthesize — deeper UI', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/synthesize');
    await expect(page.locator('main').getByRole('heading', { name: 'Text to Speech' })).toBeVisible();
  });

  // textarea placeholder
  test('textarea has correct placeholder', async ({ page }) => {
    const textarea = page.locator('textarea');
    await expect(textarea).toHaveAttribute('placeholder', 'Enter text to convert to speech...');
  });

  // label says "Text to Synthesize"
  test('label says "Text to Synthesize"', async ({ page }) => {
    await expect(page.locator('label', { hasText: 'Text to Synthesize' })).toBeVisible();
  });

  // Submit button text
  test('submit button says "Generate Speech"', async ({ page }) => {
    const btn = page.getByRole('button', { name: /Generate Speech/ });
    await expect(btn).toBeVisible();
  });

  // No audio section before generation
  test('no audio section shown before generation', async ({ page }) => {
    await expect(page.getByText('Generated Audio')).not.toBeVisible();
  });

  // Textarea has consistent height
  test('textarea has correct height class', async ({ page }) => {
    const textarea = page.locator('textarea');
    await expect(textarea).toHaveClass(/h-32/);
  });

  // Model label visible
  test('Model label is visible', async ({ page }) => {
    await expect(page.locator('label', { hasText: 'Model' })).toBeVisible();
  });

  // Form is a proper form element
  test('synthesize form submits properly', async ({ page }) => {
    const form = page.locator('form');
    await expect(form).toBeVisible();
    const submitBtn = form.locator('button[type="submit"]');
    await expect(submitBtn).toBeVisible();
  });
});

// ================================================================
// Transcribe page — deeper interaction tests
// ================================================================

test.describe('Transcribe — deeper UI', () => {
  test.beforeEach(async ({ page }) => {
    // Mock ModelSelector APIs to return empty — ensures "no model selected" state is stable
    await page.route('**/api/registry/models/transcriber', async (route) => {
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ type: 'transcriber', models: [] }) });
    });
    await page.route('**/api/cache/models/type/transcriber', async (route) => {
      await route.fulfill({ status: 200, contentType: 'application/json', body: '[]' });
    });
    await page.goto('/transcribe');
    await expect(page.locator('main').getByRole('heading', { name: 'Speech to Text' })).toBeVisible();
  });

  // "Select a model first" text when no model
  test('shows "Select a model first" when no model selected', async ({ page }) => {
    await expect(page.getByText('Select a model first')).toBeVisible();
  });

  // drop zone has reduced opacity when disabled
  test('drop zone is dimmed when no model selected', async ({ page }) => {
    const dropZone = page.locator('.border-dashed');
    await expect(dropZone).toHaveClass(/opacity-50/);
  });

  // drop zone cursor is not-allowed when disabled
  test('drop zone has cursor-not-allowed when disabled', async ({ page }) => {
    const dropZone = page.locator('.border-dashed');
    await expect(dropZone).toHaveClass(/cursor-not-allowed/);
  });

  // File input is disabled without model
  test('file input is disabled without model', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]');
    await expect(fileInput).toBeDisabled();
  });

  // Supported formats text
  test('shows supported audio formats', async ({ page }) => {
    await expect(page.getByText(/Supports WAV, MP3, M4A/)).toBeVisible();
  });

  // Model label
  test('Model label is visible', async ({ page }) => {
    await expect(page.locator('label', { hasText: 'Model' })).toBeVisible();
  });

  // No transcription result before upload
  test('no transcription result shown before upload', async ({ page }) => {
    await expect(page.getByText('Transcription')).not.toBeVisible();
  });

  // Accept attribute is audio/*
  test('file input accepts audio formats', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]');
    await expect(fileInput).toHaveAttribute('accept', 'audio/*');
  });
});

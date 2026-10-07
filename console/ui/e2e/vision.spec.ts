import { test, expect } from './fixtures/base.fixture';

// ================================================================
// Caption page tests
// ================================================================

test.describe('Caption page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/caption');
    await expect(page.locator('main').getByRole('heading', { name: /Image Captioning/ })).toBeVisible();
  });

  // [4-58] Caption mode is active by default
  test('[4-58] caption mode is active by default', async ({ page }) => {
    const captionButton = page.getByRole('button', { name: /Caption/i }).first();
    await expect(captionButton).toHaveClass(/bg-primary/);
  });

  // [4-59] VQA mode switch available
  test('[4-59] can switch to Visual QA mode', async ({ page }) => {
    const vqaButton = page.getByRole('button', { name: /Visual QA/i });
    await vqaButton.click();
    await expect(vqaButton).toHaveClass(/bg-primary/);
  });

  // [4-60] Caption image upload drop zone present
  test('[4-60] has image file upload zone', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]');
    await expect(fileInput).toHaveAttribute('accept', 'image/*');
  });

  // Model ID input
  test('has model ID input defaulting to "default"', async ({ page }) => {
    const modelInput = page.locator('input[type="text"]').first();
    const value = await modelInput.inputValue();
    expect(value).toBe('default');
  });
});

// ================================================================
// OCR page tests
// ================================================================

test.describe('OCR page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/ocr');
    await expect(page.locator('main').getByRole('heading', { name: 'Optical Character Recognition' })).toBeVisible();
  });

  // [4-67] OCR language dropdown populated from API
  test('[4-67] language dropdown is populated from API', async ({ page }) => {
    const select = page.locator('select');
    await expect(select).toBeVisible();

    // Wait for options to be populated from API
    await expect(select.locator('option')).not.toHaveCount(0, { timeout: 5_000 });
  });

  // Language text visible (not blank)
  test('language options show visible text', async ({ page }) => {
    const select = page.locator('select');
    // Wait for options to be populated
    await expect(select.locator('option')).not.toHaveCount(0, { timeout: 5_000 });

    const options = await select.locator('option').allTextContents();
    for (const opt of options) {
      expect(opt.trim()).not.toBe('');
    }

    // Should include common languages
    const hasEn = options.some(o => o.includes('en') || o.includes('English'));
    expect(hasEn).toBe(true);
  });

  // Image upload zone
  test('has image file upload zone', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]');
    await expect(fileInput).toHaveAttribute('accept', 'image/*');
  });

  // [4-68] Default OCR language is "en"
  test('[4-68] default language is "en"', async ({ page }) => {
    const select = page.locator('select');
    const value = await select.inputValue();
    expect(value).toBe('en');
  });
});

// ================================================================
// Detect page tests
// ================================================================

test.describe('Detect page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/detect');
    await expect(page.locator('main').getByRole('heading', { name: 'Object Detection' })).toBeVisible();
  });

  // [4-75] Detect model ID defaults to "default"
  test('[4-75] model ID input defaults to "default"', async ({ page }) => {
    const modelInput = page.locator('input[type="text"]').first();
    const value = await modelInput.inputValue();
    expect(value).toBe('default');
  });

  // [4-76] Confidence threshold slider with range
  test('[4-76] has confidence threshold slider with label', async ({ page }) => {
    await expect(page.getByText(/Confidence Threshold/i)).toBeVisible();
    const slider = page.locator('input[type="range"]');
    await expect(slider).toBeVisible();

    // Slider should have min/max values
    await expect(slider).toHaveAttribute('min', '0.1');
    await expect(slider).toHaveAttribute('max', '0.95');
  });

  // Image upload
  test('has image file upload zone', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]');
    await expect(fileInput).toHaveAttribute('accept', 'image/*');
  });

  // Slider value display
  test('threshold slider value is displayed as percentage', async ({ page }) => {
    // The threshold value should be shown somewhere (e.g., "50%")
    await expect(page.getByText(/%/)).toBeVisible();
  });
});

// ================================================================
// Segment page tests
// ================================================================

test.describe('Segment page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/segment');
    await expect(page.locator('main').getByRole('heading', { name: 'Image Segmentation' })).toBeVisible();
  });

  // [4-84] Segment model ID defaults to "default"
  test('[4-84] model ID input defaults to "default"', async ({ page }) => {
    const modelInput = page.locator('input[type="text"]').first();
    const value = await modelInput.inputValue();
    expect(value).toBe('default');
  });

  // Image upload
  test('has image file upload zone', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]');
    await expect(fileInput).toHaveAttribute('accept', 'image/*');

    // Drop zone with upload prompt
    const dropZone = page.locator('.border-dashed');
    await expect(dropZone).toBeVisible();
  });
});

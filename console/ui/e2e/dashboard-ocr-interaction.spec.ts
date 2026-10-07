import { test, expect } from './fixtures/base.fixture';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const TEST_IMAGE = path.resolve(__dirname, 'fixtures/test-files/test-image.png');

// ================================================================
// Dashboard — deeper interaction tests
// ================================================================

test.describe('Dashboard — deeper UI', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('main').getByRole('heading', { name: 'Dashboard' })).toBeVisible();
  });

  // Dashboard heading is rendered
  test('dashboard heading renders', async ({ page }) => {
    const heading = page.locator('main').getByRole('heading', { name: 'Dashboard' });
    await expect(heading).toHaveClass(/text-2xl/);
  });

  // RAM card shows used/total format
  test('RAM card shows used/total subtitle', async ({ page }) => {
    const ramCard = page.getByText('RAM Usage').locator('..');
    // Should show format like "X.X GB / Y.Y GB"
    await expect(ramCard.getByText(/\d+.*\/.*\d+/)).toBeVisible();
  });

  // Process Memory shows a human-readable size
  test('Process Memory shows size value', async ({ page }) => {
    const memCard = page.getByText('Process Memory').locator('..');
    // Should show MB or GB value
    await expect(memCard.getByText(/\d+.*[MG]B/)).toBeVisible();
  });

  // Cached models shows either models or empty state
  test('cached models section has proper content', async ({ page }) => {
    const section = page.getByRole('heading', { name: /Cached Models/ }).locator('..');
    // Should show either model list items or "No models cached"
    const content = section.locator('.overflow-auto, .text-muted-foreground');
    await expect(content.first()).toBeVisible();
  });

  // Loaded models shows either models or empty state
  test('loaded models section has proper content', async ({ page }) => {
    const section = page.getByRole('heading', { name: /Loaded Models/ }).locator('..');
    const content = section.locator('.overflow-auto, .text-muted-foreground');
    await expect(content.first()).toBeVisible();
  });

  // CPU Usage shows a percentage number
  test('CPU Usage card shows a numeric percentage', async ({ page }) => {
    const cpuCard = page.getByText('CPU Usage').locator('..');
    // Should have a percentage value
    await expect(cpuCard.getByText(/\d+(\.\d+)?%/)).toBeVisible();
  });
});

// ================================================================
// OCR page — deeper interaction tests
// ================================================================

test.describe('OCR — deeper UI', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/ocr');
    await expect(page.locator('main').getByRole('heading', { name: 'Optical Character Recognition' })).toBeVisible();
  });

  // fallback languages when API fails
  test('language dropdown has fallback options', async ({ page }) => {
    const select = page.locator('select');
    await expect(select).toBeVisible();

    // Should have at least the 4 fallback options
    const optionCount = await select.locator('option').count();
    expect(optionCount).toBeGreaterThanOrEqual(4);
  });

  // Language label
  test('Language label is visible', async ({ page }) => {
    await expect(page.locator('label', { hasText: 'Language' })).toBeVisible();
  });

  // Default language is "en"
  test('default language is English', async ({ page }) => {
    const select = page.locator('select');
    const value = await select.inputValue();
    expect(value).toBe('en');
  });

  // Language can be changed
  test('language selection is changeable', async ({ page }) => {
    const select = page.locator('select');
    // Wait for language options to load (more than just "en")
    await expect(select.locator('option')).not.toHaveCount(0, { timeout: 5_000 });

    const options = await select.locator('option').allTextContents();
    // Pick the second option if available
    if (options.length > 1) {
      const secondOption = await select.locator('option').nth(1).getAttribute('value');
      if (secondOption) {
        await select.selectOption(secondOption);
        const value = await select.inputValue();
        expect(value).toBe(secondOption);
      }
    }
  });

  // Upload triggers OCR
  test('uploading image triggers OCR automatically', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_IMAGE);

    // Should show loading, preview, or error
    const preview = page.locator('img[alt="Preview"]');
    const errorDiv = page.locator('.bg-destructive\\/10');
    const spinner = page.locator('.animate-spin');

    await expect(preview.or(errorDiv).or(spinner).first()).toBeVisible({ timeout: 15_000 });
  });

  // No result section before upload
  test('no result section shown before upload', async ({ page }) => {
    await expect(page.getByText('Recognized Text')).not.toBeVisible();
  });

  // Drop zone format hint
  test('drop zone shows format hints', async ({ page }) => {
    await expect(page.getByText(/Supports JPG, PNG, WebP/)).toBeVisible();
  });
});

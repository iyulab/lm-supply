import { test, expect } from './fixtures/base.fixture';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const TEST_IMAGE = path.resolve(__dirname, 'fixtures/test-files/test-image.png');

// ================================================================
// Caption page — VQA interaction tests
// ================================================================

test.describe('Caption — mode switching', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/caption');
    await expect(page.locator('main').getByRole('heading', { name: /Image Captioning/ })).toBeVisible();
  });

  // question input appears in VQA mode
  test('VQA mode shows question input after image upload', async ({ page }) => {
    // Switch to VQA mode
    await page.getByRole('button', { name: /Visual QA/i }).click();

    // Upload image
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_IMAGE);

    // Wait for preview
    await expect(page.locator('img[alt="Preview"]')).toBeVisible({ timeout: 10_000 });

    // Question input should appear
    const questionInput = page.getByPlaceholder(/Ask a question about the image/);
    await expect(questionInput).toBeVisible();

    // Ask button should be present
    const askButton = page.getByRole('button', { name: 'Ask' });
    await expect(askButton).toBeVisible();
  });

  // question input NOT visible without image in VQA mode
  test('VQA mode does not show question input before image upload', async ({ page }) => {
    // Switch to VQA mode
    await page.getByRole('button', { name: /Visual QA/i }).click();

    // No question input visible without uploaded image
    const questionInput = page.getByPlaceholder(/Ask a question about the image/);
    await expect(questionInput).not.toBeVisible();
  });

  // VQA mode does not auto-trigger inference
  test('VQA mode does not auto-caption on image upload', async ({ page }) => {
    // Switch to VQA mode first
    await page.getByRole('button', { name: /Visual QA/i }).click();

    // Upload image
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_IMAGE);

    // Preview should appear
    await expect(page.locator('img[alt="Preview"]')).toBeVisible({ timeout: 10_000 });

    // Should NOT show any loading spinner or result (no auto-inference)
    await expect(page.locator('.animate-spin')).not.toBeVisible();
  });

  // Caption mode: switching modes preserves preview
  test('[6-02] switching from Caption to VQA preserves image preview', async ({ page }) => {
    // Upload image in Caption mode
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_IMAGE);
    await expect(page.locator('img[alt="Preview"]')).toBeVisible({ timeout: 10_000 });

    // Switch to VQA
    await page.getByRole('button', { name: /Visual QA/i }).click();

    // Preview should still be visible
    await expect(page.locator('img[alt="Preview"]')).toBeVisible();
  });

  // Caption mode: mode button styling reflects active state correctly
  test('mode buttons toggle active styling', async ({ page }) => {
    const captionBtn = page.getByRole('button', { name: /Caption/i }).first();
    const vqaBtn = page.getByRole('button', { name: /Visual QA/i });

    // Initially Caption is active
    await expect(captionBtn).toHaveClass(/bg-primary/);
    await expect(vqaBtn).not.toHaveClass(/bg-primary/);

    // Switch to VQA
    await vqaBtn.click();
    await expect(vqaBtn).toHaveClass(/bg-primary/);
    await expect(captionBtn).not.toHaveClass(/bg-primary/);

    // Switch back to Caption
    await captionBtn.click();
    await expect(captionBtn).toHaveClass(/bg-primary/);
    await expect(vqaBtn).not.toHaveClass(/bg-primary/);
  });
});

test.describe('Caption — VQA form behavior', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/caption');
    await expect(page.locator('main').getByRole('heading', { name: /Image Captioning/ })).toBeVisible();

    // Switch to VQA mode and upload image
    await page.getByRole('button', { name: /Visual QA/i }).click();
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_IMAGE);
    await expect(page.locator('img[alt="Preview"]')).toBeVisible({ timeout: 10_000 });
  });

  // Ask button disabled when question is empty
  test('Ask button is disabled when question is empty', async ({ page }) => {
    const askButton = page.getByRole('button', { name: 'Ask' });
    await expect(askButton).toBeDisabled();
  });

  // Ask button enables when question is typed
  test('Ask button enables when question is entered', async ({ page }) => {
    const questionInput = page.getByPlaceholder(/Ask a question about the image/);
    await questionInput.fill('What is in this image?');

    const askButton = page.getByRole('button', { name: 'Ask' });
    await expect(askButton).toBeEnabled();
  });

  // Question input has correct placeholder
  test('question input has placeholder text', async ({ page }) => {
    const questionInput = page.getByPlaceholder(/Ask a question about the image/);
    await expect(questionInput).toBeVisible();
    await expect(questionInput).toHaveAttribute('placeholder', 'Ask a question about the image...');
  });
});

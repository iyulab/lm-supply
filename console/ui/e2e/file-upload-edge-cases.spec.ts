import { test, expect } from './fixtures/base.fixture';
import { mockJsonEndpoint, mockJsonError, mockCaptionResponse, mockOcrResponse } from './fixtures/api-mocks';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const TEST_IMAGE = path.resolve(__dirname, 'fixtures/test-files/test-image.png');

// ================================================================
// File upload edge cases
// Covers: empty file, wrong format as image, image formats, audio formats
// ================================================================

test.describe('Empty file upload', () => {
  // [7-11] Empty (0-byte) file upload — app should not crash
  test('[7-11] uploading empty audio file does not crash the app', async ({ page }) => {
    // Mock transcriber models to enable file input
    await page.route('**/api/registry/models/transcriber', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          type: 'transcriber',
          displayName: 'Transcriber',
          description: 'Speech-to-text models',
          models: [{ aliasName: 'default', repoId: 'openai/whisper-tiny', description: 'Tiny', isCached: true }],
        }),
      });
    });
    await page.route('**/api/cache/models/type/transcriber', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([{ repoId: 'openai/whisper-tiny', sizeMB: 145 }]),
      });
    });
    await page.route('**/v1/audio/transcriptions', async (route) => {
      await route.fulfill({
        status: 400,
        contentType: 'application/json',
        body: JSON.stringify({ error: { message: 'Empty or invalid audio file' } }),
      });
    });

    await page.goto('/transcribe');
    await expect(page.locator('main').getByRole('heading', { name: 'Speech to Text' })).toBeVisible();

    const fileInput = page.locator('input[type="file"]');
    await expect(fileInput).toBeEnabled({ timeout: 10_000 });

    // Upload a 0-byte file — browser may ignore or send it
    await fileInput.setInputFiles({
      name: 'empty.wav',
      mimeType: 'audio/wav',
      buffer: Buffer.alloc(0),
    });

    // Verify the page did not crash — heading should still be visible
    await expect(page.locator('main').getByRole('heading', { name: 'Speech to Text' })).toBeVisible();
    // No unhandled JS errors (Playwright would fail the test on uncaught exceptions)
  });

  // [7-11] Empty file upload to Caption — app should not crash
  test('[7-11] uploading empty image file does not crash the app', async ({ page }) => {
    await page.route('**/v1/images/caption', async (route) => {
      await route.fulfill({
        status: 400,
        contentType: 'application/json',
        body: JSON.stringify({ error: { message: 'Invalid or empty image file' } }),
      });
    });

    await page.goto('/caption');
    await expect(page.locator('main').getByRole('heading', { name: /Image Captioning/ })).toBeVisible();

    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles({
      name: 'empty.png',
      mimeType: 'image/png',
      buffer: Buffer.alloc(0),
    });

    // Verify the page did not crash
    await expect(page.locator('main').getByRole('heading', { name: /Image Captioning/ })).toBeVisible();
  });
});

test.describe('Wrong file format — image upload', () => {
  // [7-13] Text file uploaded as image to Caption
  test('[7-13] uploading text file as image shows error', async ({ page }) => {
    await page.route('**/v1/images/caption', async (route) => {
      await route.fulfill({
        status: 400,
        contentType: 'application/json',
        body: JSON.stringify({ error: { message: 'Invalid image format' } }),
      });
    });

    await page.goto('/caption');
    await expect(page.locator('main').getByRole('heading', { name: /Image Captioning/ })).toBeVisible();

    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles({
      name: 'document.txt',
      mimeType: 'text/plain',
      buffer: Buffer.from('This is a text document, not an image'),
    });

    // Should show error message
    await expect(page.getByText(/invalid|error|format/i)).toBeVisible({ timeout: 10_000 });
  });

  // [7-13] Text file uploaded as image to OCR
  test('[7-13] uploading text file as image to OCR shows error', async ({ page }) => {
    await page.route('**/v1/images/ocr', async (route) => {
      await route.fulfill({
        status: 400,
        contentType: 'application/json',
        body: JSON.stringify({ error: { message: 'Invalid image format' } }),
      });
    });

    await page.goto('/ocr');
    await expect(page.locator('main').getByRole('heading', { name: 'Optical Character Recognition' })).toBeVisible();

    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles({
      name: 'readme.txt',
      mimeType: 'text/plain',
      buffer: Buffer.from('This is not an image'),
    });

    // Should show error
    await expect(page.getByText(/invalid|error|format/i)).toBeVisible({ timeout: 10_000 });
  });
});

test.describe('Image format acceptance', () => {
  // [7-15] Image file input accepts image/*
  test('[7-15] Caption file input accepts image/* (PNG, JPG, WebP, BMP)', async ({ page }) => {
    await page.goto('/caption');
    await expect(page.locator('main').getByRole('heading', { name: /Image Captioning/ })).toBeVisible();

    const fileInput = page.locator('input[type="file"]');
    await expect(fileInput).toHaveAttribute('accept', 'image/*');
  });

  test('[7-15] OCR file input accepts image/*', async ({ page }) => {
    await page.goto('/ocr');
    await expect(page.locator('main').getByRole('heading', { name: 'Optical Character Recognition' })).toBeVisible();

    const fileInput = page.locator('input[type="file"]');
    await expect(fileInput).toHaveAttribute('accept', 'image/*');
  });

  test('[7-15] Detect file input accepts image/*', async ({ page }) => {
    await page.goto('/detect');
    await expect(page.locator('main').getByRole('heading', { name: 'Object Detection' })).toBeVisible();

    const fileInput = page.locator('input[type="file"]');
    await expect(fileInput).toHaveAttribute('accept', 'image/*');
  });
});

test.describe('Audio format acceptance', () => {
  // [7-16] Audio file input accepts audio/*
  test('[7-16] Transcribe file input accepts audio/*', async ({ page }) => {
    await page.goto('/transcribe');
    await expect(page.locator('main').getByRole('heading', { name: 'Speech to Text' })).toBeVisible();

    const fileInput = page.locator('input[type="file"]');
    await expect(fileInput).toHaveAttribute('accept', 'audio/*');
  });
});

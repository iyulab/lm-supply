import { test, expect } from './fixtures/base.fixture';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const TEST_IMAGE = path.resolve(__dirname, 'fixtures/test-files/test-image.png');
const TEST_AUDIO = path.resolve(__dirname, 'fixtures/test-files/test-audio.wav');

// ================================================================
// File upload UI behavior — vision pages
// ================================================================

test.describe('File upload — Caption page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/caption');
    await expect(page.locator('main').getByRole('heading', { name: /Image Captioning/ })).toBeVisible();
  });

  test('drop zone shows upload prompt initially', async ({ page }) => {
    await expect(page.getByText('Click to upload image')).toBeVisible();
    await expect(page.getByText(/Supports JPG, PNG, WebP/)).toBeVisible();
  });

  test('file input triggers on drop zone click', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]');
    await expect(fileInput).toHaveAttribute('accept', 'image/*');
  });

  // image preview on upload
  test('[4-61] uploading an image shows preview', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_IMAGE);

    // Preview image should appear
    const preview = page.locator('img[alt="Preview"]');
    await expect(preview).toBeVisible({ timeout: 10_000 });
  });
});

test.describe('File upload — OCR page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/ocr');
    await expect(page.locator('main').getByRole('heading', { name: 'Optical Character Recognition' })).toBeVisible();
  });

  test('drop zone shows upload prompt', async ({ page }) => {
    await expect(page.getByText('Click to upload image')).toBeVisible();
  });

  test('uploading an image shows preview', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_IMAGE);

    const preview = page.locator('img[alt="Preview"]');
    await expect(preview).toBeVisible({ timeout: 10_000 });
  });
});

test.describe('File upload — Detect page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/detect');
    await expect(page.locator('main').getByRole('heading', { name: 'Object Detection' })).toBeVisible();
  });

  test('drop zone shows upload prompt with format hints', async ({ page }) => {
    await expect(page.getByText('Click to upload image')).toBeVisible();
    await expect(page.getByText(/Supports JPG, PNG, WebP/)).toBeVisible();
  });

  test('uploading an image shows preview', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_IMAGE);

    const preview = page.locator('img[alt="Preview"]');
    await expect(preview).toBeVisible({ timeout: 10_000 });
  });
});

test.describe('File upload — Segment page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/segment');
    await expect(page.locator('main').getByRole('heading', { name: 'Image Segmentation' })).toBeVisible();
  });

  test('drop zone shows upload prompt with format hints', async ({ page }) => {
    await expect(page.getByText('Click to upload image')).toBeVisible();
    await expect(page.getByText(/Supports JPG, PNG, WebP/)).toBeVisible();
  });

  test('uploading an image shows preview', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_IMAGE);

    const preview = page.locator('img[alt="Preview"]');
    await expect(preview).toBeVisible({ timeout: 10_000 });
  });
});

// ================================================================
// File upload UI behavior — audio pages
// ================================================================

test.describe('File upload — Transcribe page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/transcribe');
    await expect(page.locator('main').getByRole('heading', { name: 'Speech to Text' })).toBeVisible();
  });

  test('drop zone shows upload prompt', async ({ page }) => {
    const dropZone = page.locator('.border-dashed');
    await expect(dropZone).toBeVisible();
    // Shows "Select a model first" or "Click to upload audio file" depending on model state
    await expect(page.getByText(/Click to upload audio file|Select a model first/)).toBeVisible();
  });

  test('audio file input accepts audio/*', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]');
    await expect(fileInput).toHaveAttribute('accept', 'audio/*');
  });
});

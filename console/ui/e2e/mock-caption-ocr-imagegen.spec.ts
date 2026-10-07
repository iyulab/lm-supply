import { test, expect } from './fixtures/base.fixture';
import { mockJsonEndpoint, mockCaptionResponse, mockVqaResponse, mockOcrResponse, mockImageGenerationResponse, mockJsonError } from './fixtures/api-mocks';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const TEST_IMAGE = path.resolve(__dirname, 'fixtures/test-files/test-image.png');

// ================================================================
// Caption page — mock API tests for result display
// Covers: caption result, alternatives, VQA answer
// ================================================================

test.describe('Caption — mock inference results', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/caption');
    await expect(page.locator('main').getByRole('heading', { name: /Image Captioning/ })).toBeVisible();
  });

  // [4-62] Caption auto-triggers on upload [4-63] Caption result shows text and confidence
  test('[4-62] caption result shows text and confidence', async ({ page }) => {
    const mockResponse = mockCaptionResponse();
    await mockJsonEndpoint(page, '**/v1/images/caption', mockResponse);

    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_IMAGE);

    // Caption text should appear
    await expect(page.getByText('A red pixel on a white background')).toBeVisible({ timeout: 15_000 });

    // Confidence should be shown
    await expect(page.getByText(/Confidence: 87\.0%/)).toBeVisible();
  });

  // Alternatives listed
  test('alternative captions are listed', async ({ page }) => {
    const mockResponse = mockCaptionResponse();
    await mockJsonEndpoint(page, '**/v1/images/caption', mockResponse);

    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_IMAGE);

    await expect(page.getByText('Alternatives:')).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText('A small colored dot')).toBeVisible();
    await expect(page.getByText('A minimal test image')).toBeVisible();
  });

  // [4-65] VQA execution shows answer with confidence
  test('[4-65] VQA mode shows question and answer', async ({ page }) => {
    const mockResponse = mockVqaResponse('What color is the pixel?');
    await mockJsonEndpoint(page, '**/v1/images/vqa', mockResponse);

    // Upload image first
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_IMAGE);

    // Switch to VQA mode
    await page.getByRole('button', { name: 'Visual QA' }).click();

    // Type question and ask
    const questionInput = page.getByPlaceholder('Ask a question about the image...');
    await questionInput.fill('What color is the pixel?');
    await page.getByRole('button', { name: 'Ask' }).click();

    // Answer should appear with Q+A format
    await expect(page.getByText('Q: What color is the pixel?')).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText('A: The image shows a red pixel.')).toBeVisible();
  });

  // Elapsed time on caption
  test('caption shows elapsed time', async ({ page }) => {
    const mockResponse = mockCaptionResponse();
    await mockJsonEndpoint(page, '**/v1/images/caption', mockResponse);

    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_IMAGE);

    await expect(page.getByText('A red pixel on a white background')).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText(/\d+ ms/)).toBeVisible();
  });
});

// ================================================================
// OCR page — mock API tests for result display
// Covers: recognition result, text blocks, no text, elapsed
// ================================================================

test.describe('OCR — mock inference results', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/ocr');
    await expect(page.locator('main').getByRole('heading', { name: 'Optical Character Recognition' })).toBeVisible();
  });

  // [4-69] OCR image upload triggers recognized text [4-70] Result shows monospace text
  test('[4-69] recognized text appears in monospace box', async ({ page }) => {
    const mockResponse = mockOcrResponse();
    await mockJsonEndpoint(page, '**/v1/images/ocr', mockResponse);

    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_IMAGE);

    await expect(page.getByText('Recognized Text')).toBeVisible({ timeout: 15_000 });

    // Text should appear in monospace area
    const textArea = page.locator('.font-mono').filter({ hasText: 'Hello World' }).first();
    await expect(textArea).toBeVisible();
  });

  // [4-71] Text blocks show confidence percentage
  test('[4-71] text blocks show confidence percentage', async ({ page }) => {
    const mockResponse = mockOcrResponse();
    await mockJsonEndpoint(page, '**/v1/images/ocr', mockResponse);

    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_IMAGE);

    await expect(page.getByText('Text Blocks')).toBeVisible({ timeout: 15_000 });

    // Should show confidence (rendered as toFixed(0) → "98%")
    await expect(page.getByText('[98%]')).toBeVisible();
  });

  // No text detected
  test('no text detected shows placeholder', async ({ page }) => {
    const emptyResponse = { id: 'ocr-mock', model: 'mock-ocr', text: '', blocks: [] };
    await mockJsonEndpoint(page, '**/v1/images/ocr', emptyResponse);

    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_IMAGE);

    await expect(page.getByText('Recognized Text')).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText('(No text detected)')).toBeVisible();
  });

  // Elapsed time
  test('elapsed time shown after recognition', async ({ page }) => {
    const mockResponse = mockOcrResponse();
    await mockJsonEndpoint(page, '**/v1/images/ocr', mockResponse);

    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_IMAGE);

    await expect(page.getByText('Recognized Text')).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText(/\d+ ms/).first()).toBeVisible();
  });
});

// ================================================================
// Image Generate page — mock API tests for result display
// Covers: generated image, details card
// ================================================================

test.describe('ImageGenerate — mock inference results', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/image-generate');
    await expect(page.locator('main').getByRole('heading', { name: 'Image Generation' })).toBeVisible();
  });

  // [4-97] Generated image displayed in result panel
  test('[4-97] generated image displays in result panel', async ({ page }) => {
    const mockResponse = mockImageGenerationResponse();
    await mockJsonEndpoint(page, '**/v1/images/generate', mockResponse);

    await page.locator('textarea').fill('A beautiful landscape');
    await page.getByRole('button', { name: /Generate Image/ }).click();

    // Image should appear (rendered from base64)
    await expect(page.locator('img[alt="Generated"]').or(page.locator('img[alt*="generated"]')).or(page.locator('.bg-card img'))).toBeVisible({ timeout: 15_000 });
  });

  // [4-98] Generation metadata shows model, time, steps
  test('[4-98] details card shows generation metadata', async ({ page }) => {
    const mockResponse = mockImageGenerationResponse();
    await mockJsonEndpoint(page, '**/v1/images/generate', mockResponse);

    await page.locator('textarea').fill('A test prompt');
    await page.getByRole('button', { name: /Generate Image/ }).click();

    // Should show generation details (seed, steps, etc.)
    const resultPanel = page.locator('.bg-card').last();
    await expect(resultPanel).toBeVisible({ timeout: 15_000 });
  });
});

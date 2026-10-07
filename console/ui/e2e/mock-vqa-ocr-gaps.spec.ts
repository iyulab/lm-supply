import { test, expect } from './fixtures/base.fixture';
import { mockJsonEndpoint, mockVqaResponse } from './fixtures/api-mocks';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const TEST_IMAGE = path.resolve(__dirname, 'fixtures/test-files/test-image.png');

// ================================================================
// VQA — Enter key and no-auto-trigger gaps
// Covers: Enter key submit, no auto-trigger in VQA mode
// ================================================================

test.describe('VQA — keyboard and auto-trigger behavior', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/caption');
    await expect(page.locator('main').getByRole('heading', { name: /Image Captioning/ })).toBeVisible();
  });

  // Enter key submits VQA question
  test('Enter key in question input submits VQA', async ({ page }) => {
    const mockResponse = mockVqaResponse('What is this?');
    await mockJsonEndpoint(page, '**/v1/images/vqa', mockResponse);

    // Upload image first
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_IMAGE);

    // Switch to VQA mode (must wait for caption auto-trigger to finish or skip)
    await page.getByRole('button', { name: 'Visual QA' }).click();

    // Type question and press Enter (not click Ask)
    const questionInput = page.getByPlaceholder('Ask a question about the image...');
    await questionInput.fill('What is this?');
    await questionInput.press('Enter');

    // Answer should appear — proving Enter triggered submission
    await expect(page.getByText('A: The image shows a red pixel.')).toBeVisible({ timeout: 15_000 });
  });

  // [4-64] VQA question input field [4-66] Image preserved on mode switch — VQA waits for user question
  test('[4-64] image upload in VQA mode does not auto-trigger inference', async ({ page }) => {
    // Switch to VQA mode BEFORE uploading
    await page.getByRole('button', { name: 'Visual QA' }).click();

    // Track whether any inference call was made
    let captionCalled = false;
    let vqaCalled = false;
    await page.route('**/v1/images/caption', async (route) => {
      captionCalled = true;
      await route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
    });
    await page.route('**/v1/images/vqa', async (route) => {
      vqaCalled = true;
      await route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
    });

    // Upload image
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_IMAGE);

    // Wait for image to be processed — question input appears but no API call should fire
    await expect(page.getByPlaceholder('Ask a question about the image...')).toBeVisible({ timeout: 5_000 });

    // Neither endpoint should have been called (VQA mode waits for user question)
    expect(captionCalled).toBe(false);
    expect(vqaCalled).toBe(false);

    // Question input should be visible (waiting for user question)
    await expect(page.getByPlaceholder('Ask a question about the image...')).toBeVisible();
  });
});

// ================================================================
// OCR — Fallback languages when API fails
// Covers: fallback languages
// ================================================================

test.describe('OCR — fallback languages', () => {
  // If OCR languages API fails, static options shown
  test('shows static fallback languages when API fails', async ({ page }) => {
    // Mock the OCR languages endpoint to fail BEFORE navigating
    await page.route('**/v1/images/ocr/languages', async (route) => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: { message: 'Service unavailable' } }),
      });
    });

    await page.goto('/ocr');
    await expect(page.locator('main').getByRole('heading', { name: 'Optical Character Recognition' })).toBeVisible();

    // Wait for fallback options to load (after API failure)
    const select = page.locator('select');
    await expect(select).toBeVisible();
    await expect(select.locator('option')).not.toHaveCount(0, { timeout: 5_000 });

    // Check for static fallback options: English (en), Korean (ko), Chinese (zh), Japanese (ja)
    // Options inside <select> are not "visible" until dropdown opens — use toHaveCount and textContent
    const options = select.locator('option');
    await expect(options).toHaveCount(4);
    await expect(select).toContainText('English (en)');
    await expect(select).toContainText('Korean (ko)');
    await expect(select).toContainText('Chinese (zh)');
    await expect(select).toContainText('Japanese (ja)');
  });
});

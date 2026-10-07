import { test, expect } from './fixtures/base.fixture';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const TEST_IMAGE = path.resolve(__dirname, 'fixtures/test-files/test-image.png');

// ================================================================
// OCR — language change and re-run
// Covers: change language → re-run OCR
// ================================================================

test.describe('OCR — language change re-run', () => {
  // [4-72] Change language and re-run OCR
  test('[4-72] changing OCR language and re-uploading produces new result', async ({ page }) => {
    let ocrCallCount = 0;
    const ocrResults = [
      { text: 'Hello World', blocks: [{ text: 'Hello World', confidence: 0.98 }], language: 'en' },
      { text: '안녕 세계', blocks: [{ text: '안녕 세계', confidence: 0.95 }], language: 'ko' },
    ];

    // Set up mock BEFORE navigation
    await page.route('**/v1/images/ocr', async (route) => {
      const result = ocrResults[Math.min(ocrCallCount, ocrResults.length - 1)];
      ocrCallCount++;
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(result),
      });
    });

    await page.goto('/ocr');
    await expect(page.locator('main').getByRole('heading', { name: 'Optical Character Recognition' })).toBeVisible();

    // First OCR with English
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_IMAGE);

    // Wait for first result — use heading as anchor
    await expect(page.getByRole('heading', { name: 'Recognized Text' })).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText('Hello World').first()).toBeVisible();

    // Change language to Korean
    const langSelect = page.locator('select').first();
    await langSelect.selectOption('ko');

    // Clear and re-upload to trigger new OCR
    await fileInput.setInputFiles([]);
    await fileInput.setInputFiles(TEST_IMAGE);

    // Wait for Korean result (use heading as anchor for timing)
    // The Korean text should replace the English text
    await expect(page.getByText(/안녕/).first()).toBeVisible({ timeout: 10_000 });

    // Verify OCR was called at least twice
    expect(ocrCallCount).toBeGreaterThanOrEqual(2);
  });
});

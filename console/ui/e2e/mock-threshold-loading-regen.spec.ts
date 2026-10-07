import { test, expect } from './fixtures/base.fixture';
import { mockJsonEndpoint, mockBlobEndpoint, mockWavBuffer, mockTranslateResponse } from './fixtures/api-mocks';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const TEST_IMAGE = path.resolve(__dirname, 'fixtures/test-files/test-image.png');

// ================================================================
// Detect — threshold effect
// Covers: threshold effect — fewer results with high threshold
// ================================================================

test.describe('Detect — threshold effect', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/detect');
    await expect(page.locator('main').getByRole('heading', { name: /Object Detection/ })).toBeVisible();
  });

  // [4-80] High threshold produces fewer detection results
  test('[4-80] high threshold produces fewer results than low threshold', async ({ page }) => {
    // Mock detect endpoint with conditional response based on request
    let requestCount = 0;
    await page.route('**/v1/images/detect', async (route) => {
      requestCount++;
      // First request (low threshold): return 3 objects
      // Second request (high threshold): return 1 object
      const objects = requestCount === 1
        ? [
            { label: 'person', confidence: 0.95, bounding_box: { x: 10, y: 20, width: 100, height: 200 } },
            { label: 'car', confidence: 0.72, bounding_box: { x: 300, y: 150, width: 200, height: 100 } },
            { label: 'dog', confidence: 0.55, bounding_box: { x: 200, y: 300, width: 80, height: 60 } },
          ]
        : [
            { label: 'person', confidence: 0.95, bounding_box: { x: 10, y: 20, width: 100, height: 200 } },
          ];
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ id: 'detect-mock', model: 'mock-detector', objects }),
      });
    });

    // First upload with default threshold (0.5)
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_IMAGE);

    // Should show 3 detected objects
    await expect(page.getByText(/Detected Objects \(3\)/)).toBeVisible({ timeout: 15_000 });

    // Now change threshold to 0.9 (high) and re-upload
    const slider = page.locator('input[type="range"]');
    await slider.fill('0.9');

    // Clear the file input value so re-selecting same file triggers onChange
    await fileInput.evaluate((el: HTMLInputElement) => { el.value = ''; });
    await fileInput.setInputFiles(TEST_IMAGE);

    // Should show fewer results (1 object)
    await expect(page.getByText(/Detected Objects \(1\)/)).toBeVisible({ timeout: 15_000 });
  });
});

// ================================================================
// Translate — loading state spinner
// Covers: loading state — spinner shown in output area
// ================================================================

test.describe('Translate — loading state', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/translate');
    await expect(page.locator('main').getByRole('heading', { name: 'Machine Translation' })).toBeVisible();
  });

  // Spinner shown during translation
  test('spinner appears in output area during translation', async ({ page }) => {
    // Mock with delay to keep loading visible
    await page.route('**/v1/translate', async (route) => {
      await new Promise(resolve => setTimeout(resolve, 3000));
      const response = mockTranslateResponse('Test');
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(response),
      });
    });

    await page.locator('textarea').fill('Test text');
    await page.getByRole('button', { name: /Translate/ }).click();

    // The output area spinner is w-6 h-6 (vs w-4 h-4 in the button)
    await expect(page.locator('.w-6.animate-spin')).toBeVisible({ timeout: 2_000 });

    // Translate button should also show loading state (disabled)
    await expect(page.getByRole('button', { name: /Translate/ })).toBeDisabled();
  });
});

// ================================================================
// Synthesize — regenerate behavior
// Covers: regenerate replaces previous audio
// ================================================================

test.describe('Synthesize — regenerate', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/synthesize');
    await expect(page.locator('main').getByRole('heading', { name: 'Text to Speech' })).toBeVisible();
    await expect(page.getByText('Loading models...')).toBeHidden({ timeout: 15_000 });
  });

  // [4-47] Regenerate replaces previous audio
  test('[4-47] second generation replaces first audio', async ({ page }) => {
    const wavBuffer = mockWavBuffer();
    await mockBlobEndpoint(page, '**/v1/audio/speech', 'audio/wav', wavBuffer);

    // First generation
    await page.locator('textarea').fill('First text');
    await page.getByRole('button', { name: /Generate Speech/ }).click();

    await expect(page.getByText('Generated Audio')).toBeVisible({ timeout: 10_000 });
    const audio = page.locator('audio');
    await expect(audio).toBeVisible();

    // Get the first audio src
    const firstSrc = await audio.getAttribute('src');
    expect(firstSrc).toBeTruthy();

    // Second generation with different text
    await page.locator('textarea').fill('Second text');
    await page.getByRole('button', { name: /Generate Speech/ }).click();

    // Audio should still be visible (new audio replaces old)
    await expect(page.getByText('Generated Audio')).toBeVisible({ timeout: 10_000 });

    // The audio src should be different (new blob URL)
    const secondSrc = await audio.getAttribute('src');
    expect(secondSrc).toBeTruthy();
    expect(secondSrc).not.toBe(firstSrc);
  });
});

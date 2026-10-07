import { test, expect } from './fixtures/base.fixture';
import { mockJsonEndpoint, mockTranslateResponse, mockBlobEndpoint, mockWavBuffer, mockJsonError } from './fixtures/api-mocks';

// ================================================================
// Translate page — mock API tests for result display
// Covers: translation result, metadata, loading
// ================================================================

test.describe('Translate — mock inference results', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/translate');
    await expect(page.locator('main').getByRole('heading', { name: 'Machine Translation' })).toBeVisible();
  });

  // [4-53] Translation result appears in output area
  test('[4-53] translation result appears in output area', async ({ page }) => {
    const mockResponse = mockTranslateResponse('Hello world');
    await mockJsonEndpoint(page, '**/v1/translate', mockResponse);

    const textarea = page.locator('textarea');
    await textarea.fill('Hello world');
    await page.getByRole('button', { name: /Translate/ }).click();

    // Translation should appear in the output div
    await expect(page.getByText('Translated: Hello world')).toBeVisible({ timeout: 10_000 });
  });

  // [4-55] Translation metadata shows model and elapsed time
  test('[4-55] translation metadata shows model and languages', async ({ page }) => {
    const mockResponse = mockTranslateResponse('Test');
    await mockJsonEndpoint(page, '**/v1/translate', mockResponse);

    await page.locator('textarea').fill('Test');
    await page.getByRole('button', { name: /Translate/ }).click();

    // Wait for result in main content area
    const main = page.locator('main');

    // Model name
    await expect(main.getByText('Model: mock-translator')).toBeVisible({ timeout: 10_000 });

    // Source and target languages
    await expect(main.getByText('Source: en')).toBeVisible();
    await expect(main.getByText('Target: ko')).toBeVisible();

    // Elapsed time
    await expect(main.getByText(/\d+ ms/)).toBeVisible();
  });

  // Error handling
  test('translation error shows error message', async ({ page }) => {
    await mockJsonError(page, '**/v1/translate', 'Translator model not available');

    await page.locator('textarea').fill('Some text');
    await page.getByRole('button', { name: /Translate/ }).click();

    await expect(page.getByText(/Translator model not available/)).toBeVisible({ timeout: 10_000 });
  });
});

// ================================================================
// Synthesize page — mock API tests for audio result
// Covers: audio player, playback, download, elapsed
// ================================================================

test.describe('Synthesize — mock audio results', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/synthesize');
    await expect(page.locator('main').getByRole('heading', { name: 'Text to Speech' })).toBeVisible();
    await expect(page.getByText('Loading models...')).toBeHidden({ timeout: 15_000 });
  });

  // [4-43] Audio player appears after speech generation
  test('[4-43] audio player appears after speech generation', async ({ page }) => {
    const wavBuffer = mockWavBuffer();
    await mockBlobEndpoint(page, '**/v1/audio/speech', 'audio/wav', wavBuffer);

    await page.locator('textarea').fill('Hello world');
    await page.getByRole('button', { name: /Generate Speech/ }).click();

    // "Generated Audio" heading should appear
    await expect(page.getByText('Generated Audio')).toBeVisible({ timeout: 10_000 });

    // Audio element should be present
    await expect(page.locator('audio')).toBeVisible();
  });

  // [4-44] Play button visible after generation
  test('[4-44] play button is visible after generation', async ({ page }) => {
    const wavBuffer = mockWavBuffer();
    await mockBlobEndpoint(page, '**/v1/audio/speech', 'audio/wav', wavBuffer);

    await page.locator('textarea').fill('Test');
    await page.getByRole('button', { name: /Generate Speech/ }).click();

    await expect(page.getByText('Generated Audio')).toBeVisible({ timeout: 10_000 });

    // Play button should be visible
    await expect(page.getByRole('button', { name: /Play/ })).toBeVisible();
  });

  // [4-45] Download WAV button visible after generation
  test('[4-45] download WAV button is visible after generation', async ({ page }) => {
    const wavBuffer = mockWavBuffer();
    await mockBlobEndpoint(page, '**/v1/audio/speech', 'audio/wav', wavBuffer);

    await page.locator('textarea').fill('Test');
    await page.getByRole('button', { name: /Generate Speech/ }).click();

    await expect(page.getByText('Generated Audio')).toBeVisible({ timeout: 10_000 });

    // Download WAV button
    await expect(page.getByRole('button', { name: /Download WAV/ })).toBeVisible();
  });

  // [4-46] Elapsed time shown after generation
  test('[4-46] elapsed time shown after generation', async ({ page }) => {
    const wavBuffer = mockWavBuffer();
    await mockBlobEndpoint(page, '**/v1/audio/speech', 'audio/wav', wavBuffer);

    await page.locator('textarea').fill('Test');
    await page.getByRole('button', { name: /Generate Speech/ }).click();

    await expect(page.getByText('Generated Audio')).toBeVisible({ timeout: 10_000 });
    await expect(page.getByText(/\d+ ms/)).toBeVisible();
  });

  // Error handling
  test('API error shows error message', async ({ page }) => {
    await mockJsonError(page, '**/v1/audio/speech', 'Synthesizer model not loaded');

    await page.locator('textarea').fill('Test');
    await page.getByRole('button', { name: /Generate Speech/ }).click();

    await expect(page.getByText(/Synthesizer model not loaded/)).toBeVisible({ timeout: 10_000 });
  });
});

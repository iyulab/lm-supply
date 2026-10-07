import { test, expect } from './fixtures/base.fixture';
import { mockJsonEndpoint, mockJsonError, mockCaptionResponse, mockModelsPageApis } from './fixtures/api-mocks';

// ================================================================
// Final coverage gap closure: 6.6, 14.3.4, 8.1.2
// ================================================================

// ================================================================
// Invalid file upload to Transcribe shows error
// ================================================================

test.describe('Transcribe — invalid file', () => {
  test('[7-12] uploading a non-audio file shows error message', async ({ page }) => {
    // Mock ModelSelector APIs so a model gets auto-selected (file input is disabled without modelId)
    await page.route('**/api/registry/models/transcriber', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          type: 'transcriber',
          displayName: 'Transcriber',
          description: 'Speech-to-text models',
          models: [{ aliasName: 'default', repoId: 'openai/whisper-tiny', description: 'Tiny Whisper', isCached: true }],
        }),
      });
    });
    await page.route('**/api/cache/models/type/transcriber', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([{ repoId: 'openai/whisper-tiny', sizeMB: 145.2 }]),
      });
    });

    // Mock the transcribe endpoint to return a 400 error (what backend does for invalid files)
    await page.route('**/v1/audio/transcriptions', async (route) => {
      await route.fulfill({
        status: 400,
        contentType: 'application/json',
        body: JSON.stringify({
          error: { message: 'Invalid file format. Expected audio file (WAV, MP3, FLAC, etc.).' },
        }),
      });
    });

    await page.goto('/transcribe');
    await expect(page.locator('main').getByRole('heading', { name: 'Speech to Text' })).toBeVisible();

    // Wait for model selector to finish loading and auto-select a model (enables file input)
    const fileInput = page.locator('input[type="file"]');
    await expect(fileInput).toBeEnabled({ timeout: 10_000 });

    // Upload a file with audio mime type but invalid content (corrupted audio)
    await fileInput.setInputFiles({
      name: 'corrupted.wav',
      mimeType: 'audio/wav',
      buffer: Buffer.from('This is not valid audio data'),
    });

    // Should show error message from the backend
    await expect(page.getByText(/Invalid file format|error/i)).toBeVisible({ timeout: 10_000 });
  });
});

// ================================================================
// Unload model via power icon
// ================================================================

test.describe('Models — unload model', () => {
  test('[3-27] clicking power icon on loaded model unloads it', async ({ page }) => {
    let unloadCalled = false;
    let unloadKey = '';

    const loadedModel = {
      modelId: 'BAAI/bge-small-en-v1.5',
      modelType: 'embedder',
      loadedAt: '2026-02-26T10:00:00.000Z',
      lastUsedAt: '2026-02-26T10:30:00.000Z',
    };

    // Mock all models page APIs with one loaded model
    await mockModelsPageApis(page, {
      cachedModels: [
        {
          repoId: 'BAAI/bge-small-en-v1.5',
          detectedType: 'Embedder',
          sizeBytes: 120_000_000,
          fileCount: 5,
          localPath: '/cache/models--BAAI--bge-small-en-v1.5',
        },
      ],
      loadedModels: [loadedModel],
    });

    // Override the loaded endpoint to track unload and return updated list
    await page.unroute('**/api/cache/loaded');
    await page.route('**/api/cache/loaded**', async (route) => {
      if (route.request().method() === 'DELETE') {
        unloadCalled = true;
        unloadKey = route.request().url().split('/cache/loaded/').pop() || '';
        await route.fulfill({ status: 200, body: '{}' });
      } else {
        // After unload, return empty list; before, return the model
        const models = unloadCalled ? [] : [loadedModel];
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(models),
        });
      }
    });

    await page.goto('/models');
    await expect(page.locator('main').getByRole('heading', { name: 'Model Management' })).toBeVisible();

    // Should show 1 loaded model with the model name in the loaded section
    await expect(page.getByText('Loaded Models (1)')).toBeVisible({ timeout: 5_000 });

    // Click the power (unload) button — exact title to avoid matching the disabled delete button
    const unloadBtn = page.locator('button[title="Unload model from memory"]');
    await expect(unloadBtn).toBeVisible();
    await unloadBtn.click();

    // Verify unload was called
    await expect.poll(() => unloadCalled, { timeout: 5_000 }).toBe(true);

    // After unload, loaded models should be empty
    await expect(page.getByText('Loaded Models (0)')).toBeVisible({ timeout: 5_000 });
  });
});

// ================================================================
// Auto-caption on image upload in caption mode
// ================================================================

test.describe('Caption — auto-trigger on upload', () => {
  test('uploading image in caption mode auto-triggers captioning', async ({ page }) => {
    let captionApiCalled = false;

    // Mock the caption endpoint
    await page.route('**/v1/images/caption', async (route) => {
      captionApiCalled = true;
      const mockResponse = mockCaptionResponse();
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockResponse),
      });
    });

    await page.goto('/caption');
    await expect(page.locator('main').getByRole('heading', { name: /Caption/ })).toBeVisible();

    // Verify we're in Caption mode (default)
    await expect(page.getByRole('button', { name: 'Caption' })).toBeVisible();

    // Upload an image — should AUTO-trigger captioning (not requiring any button click)
    const fileInput = page.locator('input[type="file"]');
    // Create a 1x1 red pixel PNG
    const pngBuffer = Buffer.from(
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==',
      'base64'
    );
    await fileInput.setInputFiles({
      name: 'test-image.png',
      mimeType: 'image/png',
      buffer: pngBuffer,
    });

    // Caption API should be called automatically (no button click needed)
    await expect(page.getByText('A red pixel on a white background')).toBeVisible({ timeout: 10_000 });
    expect(captionApiCalled).toBe(true);
  });
});

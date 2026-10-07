import { test, expect } from './fixtures/base.fixture';

// ================================================================
// Multi-domain error states: verify error display across all domains
// ================================================================

// Helper: create a 1x1 red pixel PNG buffer
const PNG_BUFFER = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==',
  'base64'
);

// Helper: create a fake WAV buffer
const WAV_BUFFER = Buffer.from('RIFF....WAVEfmt data');

// ================================================================
// 1. Text domain errors (form-submit pages)
// ================================================================

test.describe('Error states — Chat', () => {
  test('chat API error shows error message', async ({ page }) => {
    await page.route('**/v1/chat/completions', async (route) => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: { message: 'Model not loaded' } }),
      });
    });

    await page.goto('/chat');
    await expect(page.locator('main').getByRole('heading', { name: 'Chat' })).toBeVisible();
    await expect(page.getByText('Loading models...')).toBeHidden({ timeout: 15_000 });

    const input = page.locator('input[type="text"]').first();
    await input.fill('Hello');
    await input.press('Enter');

    await expect(page.getByText(/error|Model not loaded/i)).toBeVisible({ timeout: 10_000 });
  });
});

test.describe('Error states — Embed', () => {
  test('embed API error shows error message', async ({ page }) => {
    await page.route('**/v1/embeddings', async (route) => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: { message: 'Embedding model failed' } }),
      });
    });

    await page.goto('/embed');
    await expect(page.locator('main').getByRole('heading', { name: 'Embed' })).toBeVisible();

    await page.locator('textarea').fill('test text');
    await page.locator('button[type="submit"]').click();

    await expect(page.getByText(/error|fail/i)).toBeVisible({ timeout: 10_000 });
  });
});

test.describe('Error states — Rerank', () => {
  test('rerank API error shows error message', async ({ page }) => {
    await page.route('**/v1/rerank', async (route) => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: { message: 'Reranker model failed' } }),
      });
    });

    await page.goto('/rerank');
    await expect(page.locator('main').getByRole('heading', { name: 'Document Reranking' })).toBeVisible();

    await page.locator('input[type="text"]').first().fill('test query');
    await page.locator('textarea').fill('doc 1\ndoc 2');
    await page.locator('button[type="submit"]').click();

    await expect(page.getByText(/error|fail/i)).toBeVisible({ timeout: 10_000 });
  });
});

test.describe('Error states — Translate', () => {
  test('translate API error shows error message', async ({ page }) => {
    await page.route('**/v1/translate', async (route) => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: { message: 'Translation failed' } }),
      });
    });

    await page.goto('/translate');
    await expect(page.locator('main').getByRole('heading', { name: 'Machine Translation' })).toBeVisible();

    await page.locator('textarea').fill('Hello world');
    await page.locator('button[type="submit"]').click();

    await expect(page.getByText(/error|fail/i)).toBeVisible({ timeout: 10_000 });
  });
});

test.describe('Error states — Synthesize', () => {
  test('synthesize API error shows error message', async ({ page }) => {
    await page.route('**/v1/audio/speech', async (route) => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: { message: 'Synthesis failed' } }),
      });
    });

    await page.goto('/synthesize');
    await expect(page.locator('main').getByRole('heading', { name: 'Text to Speech' })).toBeVisible();

    await page.locator('textarea').fill('Test text');
    await page.locator('button[type="submit"]').click();

    await expect(page.getByText(/error|fail/i)).toBeVisible({ timeout: 10_000 });
  });
});

test.describe('Error states — ImageGenerate', () => {
  test('image generation API error shows error message', async ({ page }) => {
    await page.route('**/v1/images/generate', async (route) => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: { message: 'Generation failed' } }),
      });
    });

    await page.goto('/image-generate');
    await expect(page.locator('main').getByRole('heading', { name: 'Image Generation' })).toBeVisible();

    await page.locator('textarea').fill('a beautiful sunset');
    await page.getByRole('button', { name: 'Generate Image' }).click();

    await expect(page.getByText(/error|fail/i)).toBeVisible({ timeout: 10_000 });
  });
});

// ================================================================
// 2. Vision domain errors (file-upload pages)
// ================================================================

test.describe('Error states — Caption', () => {
  test('caption API error shows error message', async ({ page }) => {
    await page.route('**/v1/images/caption', async (route) => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: { message: 'Caption model failed' } }),
      });
    });

    await page.goto('/caption');
    await expect(page.locator('main').getByRole('heading', { name: /Caption/ })).toBeVisible();

    await page.locator('input[type="file"]').setInputFiles({
      name: 'test.png',
      mimeType: 'image/png',
      buffer: PNG_BUFFER,
    });

    await expect(page.getByText(/error|fail/i)).toBeVisible({ timeout: 10_000 });
  });
});

test.describe('Error states — OCR', () => {
  test('OCR API error shows error message', async ({ page }) => {
    await page.route('**/v1/images/ocr', async (route) => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: { message: 'OCR failed' } }),
      });
    });

    await page.goto('/ocr');
    await expect(page.locator('main').getByRole('heading', { name: /Optical Character Recognition|OCR/ })).toBeVisible();

    await page.locator('input[type="file"]').setInputFiles({
      name: 'test.png',
      mimeType: 'image/png',
      buffer: PNG_BUFFER,
    });

    await expect(page.getByText(/error|fail/i)).toBeVisible({ timeout: 10_000 });
  });
});

test.describe('Error states — Detect', () => {
  test('detect API error shows error message', async ({ page }) => {
    await page.route('**/v1/images/detect', async (route) => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: { message: 'Detection failed' } }),
      });
    });

    await page.goto('/detect');
    await expect(page.locator('main').getByRole('heading', { name: 'Detect' })).toBeVisible();

    await page.locator('input[type="file"]').setInputFiles({
      name: 'test.png',
      mimeType: 'image/png',
      buffer: PNG_BUFFER,
    });

    await expect(page.getByText(/error|fail/i)).toBeVisible({ timeout: 10_000 });
  });
});

test.describe('Error states — Segment', () => {
  test('segment API error shows error message', async ({ page }) => {
    await page.route('**/v1/images/segment', async (route) => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: { message: 'Segmentation failed' } }),
      });
    });

    await page.goto('/segment');
    await expect(page.locator('main').getByRole('heading', { name: 'Segment' })).toBeVisible();

    await page.locator('input[type="file"]').setInputFiles({
      name: 'test.png',
      mimeType: 'image/png',
      buffer: PNG_BUFFER,
    });

    await expect(page.getByText(/error|fail/i)).toBeVisible({ timeout: 10_000 });
  });
});

test.describe('Error states — Transcribe', () => {
  test('transcribe API error shows error message', async ({ page }) => {
    // Mock ModelSelector for Transcribe (file input requires modelId)
    await page.route('**/api/registry/models/transcriber', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          type: 'transcriber', models: [
            { aliasName: 'default', repoId: 'openai/whisper-tiny', description: 'Tiny', isCached: true },
          ],
        }),
      });
    });
    await page.route('**/api/cache/models/type/transcriber', async (route) => {
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify([{ repoId: 'openai/whisper-tiny', sizeMB: 145 }]) });
    });

    await page.route('**/v1/audio/transcriptions', async (route) => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: { message: 'Transcription failed' } }),
      });
    });

    await page.goto('/transcribe');
    await expect(page.locator('main').getByRole('heading', { name: 'Speech to Text' })).toBeVisible();

    // Wait for model to be auto-selected
    const fileInput = page.locator('input[type="file"]');
    await expect(fileInput).toBeEnabled({ timeout: 10_000 });

    await fileInput.setInputFiles({
      name: 'test.wav',
      mimeType: 'audio/wav',
      buffer: WAV_BUFFER,
    });

    await expect(page.getByText(/error|fail/i)).toBeVisible({ timeout: 10_000 });
  });
});

// ================================================================
// 3. Error recovery — verify error clears on successful retry
// ================================================================

test.describe('Error recovery — Embed', () => {
  test('error clears after successful retry', async ({ page }) => {
    let callCount = 0;

    await page.route('**/v1/embeddings', async (route) => {
      callCount++;
      if (callCount === 1) {
        // First call fails
        await route.fulfill({
          status: 500,
          contentType: 'application/json',
          body: JSON.stringify({ error: { message: 'Temporary failure' } }),
        });
      } else {
        // Second call succeeds
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            data: [{ embedding: [0.1, 0.2, 0.3], index: 0 }],
            model: 'test-model',
            usage: { prompt_tokens: 10, total_tokens: 10 },
          }),
        });
      }
    });

    await page.goto('/embed');
    await expect(page.locator('main').getByRole('heading', { name: 'Embed' })).toBeVisible();

    const textarea = page.locator('textarea');
    await textarea.fill('test text');

    // First click — should fail
    const submitBtn = page.locator('button[type="submit"]');
    await submitBtn.click();
    await expect(page.getByText(/error|fail/i)).toBeVisible({ timeout: 10_000 });

    // Retry — should succeed and clear error
    await submitBtn.click();
    await expect(page.getByText(/\[0\.1/)).toBeVisible({ timeout: 10_000 });
  });
});

// ================================================================
// Inference with uncached/unloaded model
// ================================================================

test.describe('Uncached model inference error', () => {
  test('inference with uncached model shows error', async ({ page }) => {
    // Mock embed endpoint to return model-not-found error
    await page.route('**/v1/embeddings', async (route) => {
      await route.fulfill({
        status: 404,
        contentType: 'application/json',
        body: JSON.stringify({ error: { message: 'Model not found: nonexistent/model' } }),
      });
    });

    await page.goto('/embed');
    await expect(page.locator('main').getByRole('heading', { name: /Embed|Text Embedding/ })).toBeVisible();

    await page.locator('textarea').fill('test text');
    await page.locator('button[type="submit"]').click();

    // Should show error about model not found
    await expect(page.getByText(/error|not found|fail/i)).toBeVisible({ timeout: 10_000 });
  });
});

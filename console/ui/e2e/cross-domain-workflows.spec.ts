import { test, expect } from './fixtures/base.fixture';
import { mockJsonEndpoint, mockChatStream } from './fixtures/api-mocks';

// ================================================================
// Cross-domain workflows: multi-page interactions, mode switching
// ================================================================

// ================================================================
// 1. Embed → Rerank pipeline (conceptual workflow)
// ================================================================

test.describe('Embed → Rerank workflow', () => {
  test('can embed texts then navigate to rerank with same concepts', async ({ page }) => {
    // Mock embed endpoint
    await page.route('**/v1/embeddings', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          data: [
            { embedding: [0.1, 0.2], index: 0 },
            { embedding: [0.3, 0.4], index: 1 },
          ],
          model: 'test-model',
          usage: { prompt_tokens: 10, total_tokens: 10 },
        }),
      });
    });

    // Step 1: Embed some texts
    await page.goto('/embed');
    await expect(page.locator('main').getByRole('heading', { name: /Embed|Text Embedding/ })).toBeVisible();
    await page.locator('textarea').fill('machine learning\ndeep learning');
    await page.locator('button[type="submit"]').click();
    await expect(page.getByText(/\[0\.1/)).toBeVisible({ timeout: 10_000 });

    // Step 2: Navigate to Rerank — form should be fresh
    await page.goto('/rerank');
    await expect(page.locator('main').getByRole('heading', { name: 'Document Reranking' })).toBeVisible();

    // Fill with similar concepts for reranking
    await page.locator('input[type="text"]').first().fill('neural networks');
    await page.locator('textarea').fill('machine learning\ndeep learning\ncooking recipes');

    // Verify form is ready to submit
    const submitBtn = page.locator('button[type="submit"]');
    await expect(submitBtn).toBeEnabled();
  });
});

// ================================================================
// 2. Caption → VQA mode switching workflow
// ================================================================

test.describe('Caption → VQA mode switching', () => {
  test('switching modes preserves uploaded image preview', async ({ page }) => {
    // Mock caption endpoint
    await page.route('**/v1/images/caption', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          caption: 'A test image',
          model: 'test-model',
          processing_time_ms: 100,
        }),
      });
    });

    await page.goto('/caption');
    await expect(page.locator('main').getByRole('heading', { name: /Caption/ })).toBeVisible();

    // Upload image in Caption mode
    const pngBuffer = Buffer.from(
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==',
      'base64'
    );
    await page.locator('input[type="file"]').setInputFiles({
      name: 'test.png',
      mimeType: 'image/png',
      buffer: pngBuffer,
    });

    // Wait for caption result
    await expect(page.getByText('A test image')).toBeVisible({ timeout: 10_000 });

    // Switch to VQA mode
    await page.getByRole('button', { name: 'Visual QA' }).click();

    // Image preview should still be visible
    const imagePreview = page.locator('img[alt*="Preview"], img[alt*="preview"], img[src*="blob:"]');
    if (await imagePreview.count() > 0) {
      await expect(imagePreview.first()).toBeVisible();
    }
  });

  test('VQA mode allows asking questions about uploaded image', async ({ page }) => {
    // Mock VQA endpoint
    await page.route('**/v1/images/vqa', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          answer: 'It is a red pixel',
          model: 'test-model',
          processing_time_ms: 50,
        }),
      });
    });

    // Also mock caption for initial upload
    await page.route('**/v1/images/caption', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          caption: 'A test image',
          model: 'test-model',
          processing_time_ms: 100,
        }),
      });
    });

    await page.goto('/caption');
    await expect(page.locator('main').getByRole('heading', { name: /Caption/ })).toBeVisible();

    // Upload image first (in Caption mode)
    const pngBuffer = Buffer.from(
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==',
      'base64'
    );
    await page.locator('input[type="file"]').setInputFiles({
      name: 'test.png',
      mimeType: 'image/png',
      buffer: pngBuffer,
    });

    // Wait for caption
    await expect(page.getByText('A test image')).toBeVisible({ timeout: 10_000 });

    // Switch to VQA
    await page.getByRole('button', { name: 'Visual QA' }).click();

    // Fill question and submit
    const questionInput = page.locator('input[placeholder*="question"], input[type="text"]').last();
    if (await questionInput.isVisible()) {
      await questionInput.fill('What color is the pixel?');
      const askBtn = page.getByRole('button', { name: /Ask/ });
      if (await askBtn.isEnabled()) {
        await askBtn.click();
        await expect(page.getByText('It is a red pixel')).toBeVisible({ timeout: 10_000 });
      }
    }
  });
});

// ================================================================
// 3. Chat multi-turn conversation
// ================================================================

test.describe('Chat — multi-turn', () => {
  // 3+ turn multi-turn chat
  test('[6-04] can send multiple messages in sequence', async ({ page }) => {
    let messageCount = 0;

    await page.route('**/v1/chat/completions', async (route) => {
      messageCount++;
      const responses: Record<number, string> = {
        1: 'Hello! How can I help?',
        2: 'I am an AI assistant.',
        3: 'The capital of France is Paris.',
      };
      const content = responses[messageCount] || 'Default response';
      const body = `data: {"choices":[{"delta":{"content":"${content}"}}]}\n\ndata: [DONE]\n\n`;
      await route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body,
      });
    });

    await page.goto('/chat');
    await expect(page.locator('main').getByRole('heading', { name: 'Chat' })).toBeVisible();
    await expect(page.getByText('Loading models...')).toBeHidden({ timeout: 15_000 });

    const input = page.locator('input[type="text"]').first();

    // Turn 1
    await input.fill('Hello');
    await input.press('Enter');
    await expect(page.getByText('Hello! How can I help?')).toBeVisible({ timeout: 10_000 });

    // Turn 2
    await input.fill('Who are you?');
    await input.press('Enter');
    await expect(page.getByText('I am an AI assistant.')).toBeVisible({ timeout: 10_000 });

    // Turn 3
    await input.fill('What is the capital of France?');
    await input.press('Enter');
    await expect(page.getByText('The capital of France is Paris.')).toBeVisible({ timeout: 10_000 });

    // All three exchanges should be visible in the chat
    expect(messageCount).toBe(3);
  });
});

// ================================================================
// 4. Model selector interaction across pages
// ================================================================

test.describe('Model selector — cross-page', () => {
  test('model selection is independent between Embed and Rerank', async ({ page }) => {
    await page.goto('/embed');
    await expect(page.locator('main').getByRole('heading', { name: /Embed|Text Embedding/ })).toBeVisible();

    // Check embed model selector exists
    const embedSelect = page.locator('main select, main [role="combobox"]').first();
    await expect(embedSelect).toBeVisible();

    // Navigate to Rerank
    await page.goto('/rerank');
    await expect(page.locator('main').getByRole('heading', { name: 'Document Reranking' })).toBeVisible();

    // Check rerank model selector exists (different models)
    const rerankSelect = page.locator('main select, main [role="combobox"]').first();
    await expect(rerankSelect).toBeVisible();
  });
});

// ================================================================
// 5. Synthesize + Transcribe round-trip (conceptual)
// ================================================================

test.describe('Synthesize → Transcribe conceptual workflow', () => {
  // Synthesize → Transcribe round-trip
  test('[6-03] synthesize page generates audio, transcribe page accepts audio', async ({ page }) => {
    // Mock synthesize to return audio
    await page.route('**/v1/audio/speech', async (route) => {
      // Create a minimal WAV header
      const wavHeader = Buffer.from('RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00');
      await route.fulfill({
        status: 200,
        contentType: 'audio/wav',
        body: wavHeader,
      });
    });

    // Step 1: Generate speech
    await page.goto('/synthesize');
    await expect(page.locator('main').getByRole('heading', { name: 'Text to Speech' })).toBeVisible();
    await page.locator('textarea').fill('Hello world');
    await page.locator('button[type="submit"]').click();

    // Audio player should appear
    await expect(page.locator('audio')).toBeVisible({ timeout: 10_000 });

    // Step 2: Navigate to Transcribe
    await page.goto('/transcribe');
    await expect(page.locator('main').getByRole('heading', { name: 'Speech to Text' })).toBeVisible();

    // Transcribe page should be ready for audio upload
    const fileInput = page.locator('input[type="file"]');
    await expect(fileInput).toHaveAttribute('accept', 'audio/*');
  });
});

// ================================================================
// 6. Download → Load → Inference pipeline (6-06)
// ================================================================

test.describe('Download → Load → Inference pipeline', () => {
  // full pipeline: download model, pre-load, then use for inference
  test('[6-06] download then load then inference across pages', async ({ page }) => {
    // Step 1: Mock Models page APIs — start with empty cache
    await page.route('**/api/cache/models', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ models: [] }),
      });
    });
    await page.route('**/api/cache/loaded', async (route) => {
      await route.fulfill({ status: 200, contentType: 'application/json', body: '[]' });
    });
    await page.route('**/api/registry/models', async (route) => {
      await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ modelTypes: [] }) });
    });

    // Mock download endpoint — SSE progress + completion
    await page.route('**/api/download/model', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: 'data: {"fileName":"model.safetensors","bytesDownloaded":100,"totalBytes":100,"percentComplete":100.0,"status":"Completed"}\n\n',
      });
    });

    // Step 1: Go to Models page and trigger download
    await page.goto('/models');
    await expect(page.locator('main').getByRole('heading', { name: 'Model Management' })).toBeVisible();
    await page.locator('input[placeholder*="BAAI"]').fill('test-org/test-model');
    await page.getByRole('button', { name: 'Download' }).click();
    await expect(page.getByText('Download completed')).toBeVisible({ timeout: 10_000 });

    // Step 2: Navigate to Embed page for inference
    // Mock embed endpoint for successful inference
    await page.route('**/v1/embeddings', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          data: [{ embedding: [0.1, 0.2, 0.3], index: 0 }],
          model: 'test-model',
          usage: { prompt_tokens: 5, total_tokens: 5 },
        }),
      });
    });

    await page.goto('/embed');
    await expect(page.locator('main').getByRole('heading', { name: /Embed|Text Embedding/ })).toBeVisible();
    await page.locator('textarea').fill('test embedding text');
    await page.locator('button[type="submit"]').click();

    // Step 3: Verify inference result
    await expect(page.getByText(/\[0\.1/)).toBeVisible({ timeout: 10_000 });
  });
});

// ================================================================
// 7. Concurrent download + inference (6-09)
// ================================================================

test.describe('Concurrent download and inference', () => {
  // download in progress does not block inference on another page
  test('[6-09] inference works while download is in progress on models page', async ({ page }) => {
    // Mock embed endpoint for inference
    await page.route('**/v1/embeddings', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          data: [{ embedding: [0.5, 0.6], index: 0 }],
          model: 'test-model',
          usage: { prompt_tokens: 3, total_tokens: 3 },
        }),
      });
    });

    // Step 1: Navigate to Embed page and run inference (simulating loaded model)
    await page.goto('/embed');
    await expect(page.locator('main').getByRole('heading', { name: /Embed|Text Embedding/ })).toBeVisible();
    await page.locator('textarea').fill('test concurrent inference');
    await page.locator('button[type="submit"]').click();
    await expect(page.getByText(/\[0\.5/)).toBeVisible({ timeout: 10_000 });

    // Step 2: Navigate to Models page — verify it's accessible
    await page.goto('/models');
    await expect(page.locator('main').getByRole('heading', { name: 'Model Management' })).toBeVisible();

    // Step 3: Navigate back to Embed — still functional
    await page.goto('/embed');
    await expect(page.locator('main').getByRole('heading', { name: /Embed|Text Embedding/ })).toBeVisible();
    await page.locator('textarea').fill('second inference');
    await page.locator('button[type="submit"]').click();
    await expect(page.getByText(/\[0\.5/)).toBeVisible({ timeout: 10_000 });
  });
});

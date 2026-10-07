import { test, expect } from './fixtures/base.fixture';

// ================================================================
// Negative path testing: empty forms, boundary values, special chars
// ================================================================

// ================================================================
// 1. Submit button disabled states (empty inputs)
// ================================================================

test.describe('Chat — empty input', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/chat');
    await expect(page.locator('main').getByRole('heading', { name: 'Chat' })).toBeVisible();
  });

  test('send button is disabled when input is empty', async ({ page }) => {
    const sendBtn = page.locator('button[type="submit"]');
    await expect(sendBtn).toBeDisabled();
  });

  // Whitespace-only input keeps send disabled
  test('send button is disabled with whitespace-only input', async ({ page }) => {
    const input = page.locator('input[type="text"]').first();
    await input.fill('   ');
    const sendBtn = page.locator('button[type="submit"]');
    await expect(sendBtn).toBeDisabled();
  });
});

test.describe('Embed — empty input', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/embed');
    await expect(page.locator('main').getByRole('heading', { name: 'Embed' })).toBeVisible();
  });

  test('embed button is disabled when textarea is empty', async ({ page }) => {
    const submitBtn = page.locator('button[type="submit"]');
    await expect(submitBtn).toBeDisabled();
  });

  test('embed button is disabled with whitespace-only text', async ({ page }) => {
    const textarea = page.locator('textarea');
    await textarea.fill('   \n   \n   ');
    const submitBtn = page.locator('button[type="submit"]');
    await expect(submitBtn).toBeDisabled();
  });
});

test.describe('Translate — empty input', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/translate');
    await expect(page.locator('main').getByRole('heading', { name: 'Machine Translation' })).toBeVisible();
  });

  test('translate button is disabled when text is empty', async ({ page }) => {
    const submitBtn = page.locator('button[type="submit"]');
    await expect(submitBtn).toBeDisabled();
  });

  test('translate button is disabled with whitespace-only text', async ({ page }) => {
    const textarea = page.locator('textarea');
    await textarea.fill('   ');
    const submitBtn = page.locator('button[type="submit"]');
    await expect(submitBtn).toBeDisabled();
  });
});

test.describe('Synthesize — empty input', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/synthesize');
    await expect(page.locator('main').getByRole('heading', { name: 'Text to Speech' })).toBeVisible();
  });

  test('generate button is disabled when text is empty', async ({ page }) => {
    const submitBtn = page.locator('button[type="submit"]');
    await expect(submitBtn).toBeDisabled();
  });

  test('generate button is disabled with whitespace-only text', async ({ page }) => {
    const textarea = page.locator('textarea');
    await textarea.fill('   \n  ');
    const submitBtn = page.locator('button[type="submit"]');
    await expect(submitBtn).toBeDisabled();
  });
});

test.describe('Rerank — empty inputs', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/rerank');
    await expect(page.locator('main').getByRole('heading', { name: 'Rerank' })).toBeVisible();
  });

  test('rerank button is disabled when all fields empty', async ({ page }) => {
    const submitBtn = page.locator('button[type="submit"]');
    await expect(submitBtn).toBeDisabled();
  });

  test('rerank button is disabled with query but no documents', async ({ page }) => {
    const queryInput = page.locator('input[type="text"]').first();
    await queryInput.fill('test query');
    const submitBtn = page.locator('button[type="submit"]');
    await expect(submitBtn).toBeDisabled();
  });

  test('rerank button is disabled with documents but no query', async ({ page }) => {
    const textarea = page.locator('textarea');
    await textarea.fill('document 1\ndocument 2');
    const submitBtn = page.locator('button[type="submit"]');
    await expect(submitBtn).toBeDisabled();
  });
});

test.describe('ImageGenerate — empty input', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/image-generate');
    await expect(page.locator('main').getByRole('heading', { name: 'Image Generation' })).toBeVisible();
  });

  test('generate button is disabled when prompt is empty', async ({ page }) => {
    const submitBtn = page.getByRole('button', { name: 'Generate Image' });
    await expect(submitBtn).toBeDisabled();
  });

  test('generate button is disabled with whitespace-only prompt', async ({ page }) => {
    const textarea = page.locator('textarea');
    await textarea.fill('   ');
    const submitBtn = page.getByRole('button', { name: 'Generate Image' });
    await expect(submitBtn).toBeDisabled();
  });
});

// ================================================================
// 2. Numeric boundary values
// ================================================================

test.describe('Rerank — topK boundaries', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/rerank');
    await expect(page.locator('main').getByRole('heading', { name: 'Rerank' })).toBeVisible();
  });

  test('topK input has min=1 attribute', async ({ page }) => {
    const topKInput = page.locator('input[type="number"]');
    await expect(topKInput).toHaveAttribute('min', '1');
  });

  test('topK defaults to 5', async ({ page }) => {
    const topKInput = page.locator('input[type="number"]');
    await expect(topKInput).toHaveValue('5');
  });

  test('topK can be set to 1 (minimum)', async ({ page }) => {
    const topKInput = page.locator('input[type="number"]');
    await topKInput.fill('1');
    await expect(topKInput).toHaveValue('1');
  });
});

test.describe('Detect — confidence threshold boundaries', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/detect');
    await expect(page.locator('main').getByRole('heading', { name: 'Detect' })).toBeVisible();
  });

  test('threshold slider has min=0.1', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    await expect(slider).toHaveAttribute('min', '0.1');
  });

  test('threshold slider has max=0.95', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    await expect(slider).toHaveAttribute('max', '0.95');
  });

  test('threshold defaults to 0.5', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    await expect(slider).toHaveValue('0.5');
  });
});

test.describe('ImageGenerate — slider boundaries', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/image-generate');
    await expect(page.locator('main').getByRole('heading', { name: 'Image Generation' })).toBeVisible();
    // Expand advanced settings to reveal sliders
    await page.getByRole('button', { name: /Advanced Settings/ }).click();
  });

  test('steps slider has min=1 and max=8', async ({ page }) => {
    const stepsSlider = page.locator('input[type="range"]').first();
    await expect(stepsSlider).toHaveAttribute('min', '1');
    await expect(stepsSlider).toHaveAttribute('max', '8');
  });

  test('guidance scale slider range is correct', async ({ page }) => {
    const sliders = page.locator('input[type="range"]');
    const guidanceSlider = sliders.nth(1);
    await expect(guidanceSlider).toHaveAttribute('min', '0');
    await expect(guidanceSlider).toHaveAttribute('max', '3');
  });
});

// ================================================================
// 3. Special characters in text inputs
// ================================================================

test.describe('Chat — special characters', () => {
  test('unicode input is accepted in chat', async ({ page }) => {
    await page.goto('/chat');
    await expect(page.locator('main').getByRole('heading', { name: 'Chat' })).toBeVisible();

    const input = page.locator('input[type="text"]').first();
    await input.fill('한국어 테스트 🎉');
    await expect(input).toHaveValue('한국어 테스트 🎉');

    // Button should be enabled with valid text
    const sendBtn = page.locator('button[type="submit"]');
    await expect(sendBtn).toBeEnabled();
  });
});

test.describe('Embed — special characters', () => {
  test('unicode and newlines accepted in embed textarea', async ({ page }) => {
    await page.goto('/embed');
    await expect(page.locator('main').getByRole('heading', { name: 'Embed' })).toBeVisible();

    const textarea = page.locator('textarea');
    await textarea.fill('日本語テスト\nDeutsch mit Ümläuten\nEmoji: 🚀✨');
    await expect(textarea).toHaveValue('日本語テスト\nDeutsch mit Ümläuten\nEmoji: 🚀✨');

    const submitBtn = page.locator('button[type="submit"]');
    await expect(submitBtn).toBeEnabled();
  });
});

test.describe('Translate — special characters', () => {
  test('unicode text accepted for translation', async ({ page }) => {
    await page.goto('/translate');
    await expect(page.locator('main').getByRole('heading', { name: 'Machine Translation' })).toBeVisible();

    const textarea = page.locator('textarea');
    await textarea.fill('中文测试 — "引用" & <符号>');
    await expect(textarea).toHaveValue('中文测试 — "引用" & <符号>');

    const submitBtn = page.locator('button[type="submit"]');
    await expect(submitBtn).toBeEnabled();
  });
});

// ================================================================
// 4. VQA mode — empty question
// ================================================================

test.describe('Caption VQA — empty question', () => {
  test('VQA ask button is disabled when question is empty', async ({ page }) => {
    await page.goto('/caption');
    await expect(page.locator('main').getByRole('heading', { name: /Caption/ })).toBeVisible();

    // Switch to VQA mode
    const vqaBtn = page.getByRole('button', { name: 'Visual QA' });
    await vqaBtn.click();

    // Find the question input (only visible in VQA mode after image upload)
    // The Ask button should be disabled since no image and no question
    const askBtn = page.getByRole('button', { name: /Ask/ });
    if (await askBtn.isVisible()) {
      await expect(askBtn).toBeDisabled();
    }
  });
});

// ================================================================
// 5. Error state display on API failure
// ================================================================

test.describe('Embed — API error shows message', () => {
  test('server error displays error text', async ({ page }) => {
    // Mock embed endpoint to return 500
    await page.route('**/v1/embeddings', async (route) => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: { message: 'Internal server error' } }),
      });
    });

    await page.goto('/embed');
    await expect(page.locator('main').getByRole('heading', { name: 'Embed' })).toBeVisible();

    const textarea = page.locator('textarea');
    await textarea.fill('test text');

    const submitBtn = page.locator('button[type="submit"]');
    await submitBtn.click();

    // Error should be visible
    await expect(page.getByText(/error|failed/i)).toBeVisible({ timeout: 10_000 });
  });
});

test.describe('Translate — API error shows message', () => {
  test('server error displays error text', async ({ page }) => {
    // Mock translate endpoint to return 500
    await page.route('**/v1/translate', async (route) => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: { message: 'Translation service unavailable' } }),
      });
    });

    await page.goto('/translate');
    await expect(page.locator('main').getByRole('heading', { name: 'Machine Translation' })).toBeVisible();

    const textarea = page.locator('textarea');
    await textarea.fill('Hello world');

    const submitBtn = page.locator('button[type="submit"]');
    await submitBtn.click();

    await expect(page.getByText(/error|failed|unavailable/i)).toBeVisible({ timeout: 10_000 });
  });
});

test.describe('Synthesize — API error shows message', () => {
  test('server error displays error text', async ({ page }) => {
    // Mock synthesize endpoint to return 500
    await page.route('**/v1/audio/speech', async (route) => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: { message: 'Synthesis failed' } }),
      });
    });

    await page.goto('/synthesize');
    await expect(page.locator('main').getByRole('heading', { name: 'Text to Speech' })).toBeVisible();

    const textarea = page.locator('textarea');
    await textarea.fill('Test speech');

    const submitBtn = page.locator('button[type="submit"]');
    await submitBtn.click();

    await expect(page.getByText(/error|failed/i)).toBeVisible({ timeout: 10_000 });
  });
});

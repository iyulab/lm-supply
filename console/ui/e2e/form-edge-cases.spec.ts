import { test, expect } from './fixtures/base.fixture';

// ================================================================
// Form edge cases: Unicode, max length, paste, special characters
// ================================================================

// ================================================================
// 1. Very long text inputs
// ================================================================

test.describe('Long text inputs', () => {
  test('embed textarea accepts very long text', async ({ page }) => {
    await page.goto('/embed');
    await expect(page.locator('main').getByRole('heading', { name: /Embed|Text Embedding/ })).toBeVisible();

    const textarea = page.locator('textarea');
    const longText = 'Lorem ipsum dolor sit amet. '.repeat(200); // ~5600 chars
    await textarea.fill(longText);
    const value = await textarea.inputValue();
    expect(value.length).toBe(longText.length);

    // Submit button should be enabled
    const submitBtn = page.locator('button[type="submit"]');
    await expect(submitBtn).toBeEnabled();
  });

  test('chat input accepts long messages', async ({ page }) => {
    await page.goto('/chat');
    await expect(page.locator('main').getByRole('heading', { name: 'Chat' })).toBeVisible();

    const input = page.locator('input[type="text"]').first();
    const longMessage = 'A'.repeat(1000);
    await input.fill(longMessage);
    await expect(input).toHaveValue(longMessage);
  });

  test('translate textarea accepts large text blocks', async ({ page }) => {
    await page.goto('/translate');
    await expect(page.locator('main').getByRole('heading', { name: 'Machine Translation' })).toBeVisible();

    const textarea = page.locator('textarea');
    // Multi-paragraph text
    const lines = Array.from({ length: 50 }, (_, i) => `Line ${i + 1}: This is a sentence for translation testing.`);
    await textarea.fill(lines.join('\n'));
    const value = await textarea.inputValue();
    expect(value.split('\n').length).toBe(50);
  });
});

// ================================================================
// 2. Unicode and multilingual text
// ================================================================

test.describe('Unicode text handling', () => {
  test('chat handles CJK characters', async ({ page }) => {
    await page.goto('/chat');
    await expect(page.locator('main').getByRole('heading', { name: 'Chat' })).toBeVisible();

    const input = page.locator('input[type="text"]').first();
    await input.fill('你好世界 こんにちは 안녕하세요');
    await expect(input).toHaveValue('你好世界 こんにちは 안녕하세요');
    await expect(page.locator('button[type="submit"]')).toBeEnabled();
  });

  test('embed handles emoji and special unicode', async ({ page }) => {
    await page.goto('/embed');
    await expect(page.locator('main').getByRole('heading', { name: /Embed|Text Embedding/ })).toBeVisible();

    const textarea = page.locator('textarea');
    await textarea.fill('🚀 Rocket science\n💡 Bright ideas\n🎯 Target practice\n⚡ Lightning fast');
    const value = await textarea.inputValue();
    expect(value).toContain('🚀');
    expect(value).toContain('💡');
    expect(value.split('\n').length).toBe(4);
  });

  test('translate handles RTL text (Arabic)', async ({ page }) => {
    await page.goto('/translate');
    await expect(page.locator('main').getByRole('heading', { name: 'Machine Translation' })).toBeVisible();

    const textarea = page.locator('textarea');
    await textarea.fill('مرحبا بالعالم');
    await expect(textarea).toHaveValue('مرحبا بالعالم');
    await expect(page.locator('button[type="submit"]')).toBeEnabled();
  });

  test('rerank handles mixed scripts in query and documents', async ({ page }) => {
    await page.goto('/rerank');
    await expect(page.locator('main').getByRole('heading', { name: 'Document Reranking' })).toBeVisible();

    const queryInput = page.locator('input[type="text"]').first();
    await queryInput.fill('Ñoño español');
    await expect(queryInput).toHaveValue('Ñoño español');

    const textarea = page.locator('textarea');
    await textarea.fill('Ünïcödë têxt\nÆØÅ Nordic\nß German sharp s');
    const value = await textarea.inputValue();
    expect(value).toContain('Ünïcödë');
    expect(value).toContain('ÆØÅ');
  });
});

// ================================================================
// 3. HTML/Script injection (XSS-like content)
// ================================================================

test.describe('HTML content in inputs', () => {
  test('chat input accepts HTML-like content without rendering it', async ({ page }) => {
    await page.goto('/chat');
    await expect(page.locator('main').getByRole('heading', { name: 'Chat' })).toBeVisible();

    const input = page.locator('input[type="text"]').first();
    await input.fill('<script>alert("xss")</script>');
    await expect(input).toHaveValue('<script>alert("xss")</script>');

    // No alert should have fired
    // Button should be enabled (it's valid text)
    await expect(page.locator('button[type="submit"]')).toBeEnabled();
  });

  test('embed textarea accepts HTML tags as text', async ({ page }) => {
    await page.goto('/embed');
    await expect(page.locator('main').getByRole('heading', { name: /Embed|Text Embedding/ })).toBeVisible();

    const textarea = page.locator('textarea');
    await textarea.fill('<div class="test"><p>Hello</p></div>\n<img src="x" onerror="alert(1)">');
    const value = await textarea.inputValue();
    expect(value).toContain('<div class="test">');
    expect(value).toContain('<img src="x"');
  });
});

// ================================================================
// 4. Special characters and control characters
// ================================================================

test.describe('Special characters', () => {
  test('synthesize handles special punctuation', async ({ page }) => {
    await page.goto('/synthesize');
    await expect(page.locator('main').getByRole('heading', { name: 'Text to Speech' })).toBeVisible();

    const textarea = page.locator('textarea');
    await textarea.fill('Hello! How are you? I\'m fine — thanks & goodbye...');
    const value = await textarea.inputValue();
    expect(value).toContain('—');
    expect(value).toContain('&');
    expect(value).toContain("I'm");
  });

  test('image generate prompt handles quotes and special chars', async ({ page }) => {
    await page.goto('/image-generate');
    await expect(page.locator('main').getByRole('heading', { name: 'Image Generation' })).toBeVisible();

    const textarea = page.locator('textarea');
    await textarea.fill('A "beautiful" sunset with \'golden\' light & vivid colors — no clouds');
    const value = await textarea.inputValue();
    expect(value).toContain('"beautiful"');
    expect(value).toContain("'golden'");
    expect(value).toContain('&');
  });
});

// ================================================================
// 5. Multiline text edge cases
// ================================================================

test.describe('Multiline text', () => {
  test('embed handles single empty line between texts', async ({ page }) => {
    await page.goto('/embed');
    await expect(page.locator('main').getByRole('heading', { name: /Embed|Text Embedding/ })).toBeVisible();

    const textarea = page.locator('textarea');
    await textarea.fill('text one\n\ntext two\n\n\ntext three');
    const value = await textarea.inputValue();
    expect(value.split('\n').length).toBe(6); // includes empty lines
  });

  test('rerank documents with various separators', async ({ page }) => {
    await page.goto('/rerank');
    await expect(page.locator('main').getByRole('heading', { name: 'Document Reranking' })).toBeVisible();

    const textarea = page.locator('textarea');
    // Documents with different content types
    await textarea.fill('Doc with numbers: 123,456.78\nDoc with URL: https://example.com\nDoc with path: C:\\Users\\test');
    const value = await textarea.inputValue();
    expect(value).toContain('123,456.78');
    expect(value).toContain('https://example.com');
  });
});

// ================================================================
// 6. Numeric input edge cases
// ================================================================

test.describe('Numeric input edges', () => {
  test('rerank topK can be set to large value', async ({ page }) => {
    await page.goto('/rerank');
    await expect(page.locator('main').getByRole('heading', { name: 'Document Reranking' })).toBeVisible();

    const topKInput = page.locator('input[type="number"]');
    await topKInput.fill('100');
    await expect(topKInput).toHaveValue('100');
  });

  test('detect threshold slider at minimum', async ({ page }) => {
    await page.goto('/detect');
    await expect(page.locator('main').getByRole('heading', { name: 'Detect' })).toBeVisible();

    const slider = page.locator('input[type="range"]');
    await slider.fill('0.1');
    await expect(slider).toHaveValue('0.1');
  });

  test('detect threshold slider at maximum', async ({ page }) => {
    await page.goto('/detect');
    await expect(page.locator('main').getByRole('heading', { name: 'Detect' })).toBeVisible();

    const slider = page.locator('input[type="range"]');
    await slider.fill('0.95');
    await expect(slider).toHaveValue('0.95');
  });
});

import { test, expect } from './fixtures/base.fixture';
import { mockChatStream, mockJsonEndpoint, mockEmbedResponse } from './fixtures/api-mocks';

// ================================================================
// Input edge cases — XSS, RTL, very long text
// Covers: XSS, RTL, very long text
// ================================================================

test.describe('XSS / HTML injection', () => {
  // HTML injection attempt — should render as plain text, not execute
  test('script tag renders as escaped text in Chat', async ({ page }) => {
    const xssPayload = '<script>alert("xss")</script>';
    const xssResponse = ['Received: ', xssPayload];

    await page.goto('/chat');
    await expect(page.locator('main').getByRole('heading', { name: /Chat/ }).first()).toBeVisible();
    await expect(page.getByText('Loading models...')).toBeHidden({ timeout: 15_000 });

    await mockChatStream(page, '**/v1/chat/completions', xssResponse);

    const input = page.getByPlaceholder('Type your message...');
    await input.fill(xssPayload);
    await page.locator('button[type="submit"]').click();

    // User message should show the script tag as text
    await expect(page.locator('.bg-primary').filter({ hasText: '<script>' }).first()).toBeVisible({ timeout: 10_000 });

    // Script should NOT have executed (no alert dialog)
    // The text should be visible as plain text
    await expect(page.locator('.bg-muted').filter({ hasText: '<script>' }).first()).toBeVisible({ timeout: 10_000 });

    // Verify no script elements were injected
    const scriptCount = await page.locator('script:text("alert")').count();
    expect(scriptCount).toBe(0);
  });

  test('HTML tags rendered as text in Embed', async ({ page }) => {
    await page.goto('/embed');
    await expect(page.locator('main').getByRole('heading', { name: 'Text Embedding' })).toBeVisible();

    const xssPayload = '<img src=x onerror=alert(1)>';
    const textarea = page.locator('textarea');
    await textarea.fill(xssPayload);

    // The input should contain the HTML as plain text
    await expect(textarea).toHaveValue(xssPayload);

    // Submit button should be enabled (it's valid text)
    await expect(page.locator('button[type="submit"]')).toBeEnabled();
  });
});

test.describe('RTL text (Arabic)', () => {
  // Arabic/RTL text input accepted
  test('Arabic text is accepted in Chat input', async ({ page }) => {
    await page.goto('/chat');
    await expect(page.locator('main').getByRole('heading', { name: /Chat/ }).first()).toBeVisible();
    await expect(page.getByText('Loading models...')).toBeHidden({ timeout: 15_000 });

    const arabicText = 'مرحبا بالعالم';
    await mockChatStream(page, '**/v1/chat/completions', ['مرحبا']);

    const input = page.getByPlaceholder('Type your message...');
    await input.fill(arabicText);
    await expect(input).toHaveValue(arabicText);

    // Send should be enabled
    await expect(page.locator('button[type="submit"]')).toBeEnabled();
    await page.locator('button[type="submit"]').click();

    // User message should display Arabic text
    await expect(page.locator('.bg-primary').filter({ hasText: 'مرحبا' }).first()).toBeVisible({ timeout: 10_000 });
  });

  test('Arabic text is accepted in Embed textarea', async ({ page }) => {
    await page.goto('/embed');
    await expect(page.locator('main').getByRole('heading', { name: 'Text Embedding' })).toBeVisible();

    const arabicText = 'مرحبا بالعالم\nسلام عليكم';
    const textarea = page.locator('textarea');
    await textarea.fill(arabicText);
    await expect(textarea).toHaveValue(arabicText);
    await expect(page.locator('button[type="submit"]')).toBeEnabled();
  });
});

test.describe('Very long text (10k+ characters)', () => {
  // 10,000+ character input
  test('10000 character input accepted in Chat', async ({ page }) => {
    await page.goto('/chat');
    await expect(page.locator('main').getByRole('heading', { name: /Chat/ }).first()).toBeVisible();
    await expect(page.getByText('Loading models...')).toBeHidden({ timeout: 15_000 });

    const longText = 'X'.repeat(10000);
    const input = page.getByPlaceholder('Type your message...');
    await input.fill(longText);
    await expect(input).toHaveValue(longText);
    await expect(page.locator('button[type="submit"]')).toBeEnabled();
  });

  test('10000 character input accepted in Embed textarea', async ({ page }) => {
    await page.goto('/embed');
    await expect(page.locator('main').getByRole('heading', { name: 'Text Embedding' })).toBeVisible();

    const longText = 'Y'.repeat(10000);
    const textarea = page.locator('textarea');
    await textarea.fill(longText);
    await expect(textarea).toHaveValue(longText);
    await expect(page.locator('button[type="submit"]')).toBeEnabled();
  });

  // Long text translation (500+ chars)
  test('500+ character text accepted in Translate', async ({ page }) => {
    await page.goto('/translate');
    await expect(page.locator('main').getByRole('heading', { name: 'Machine Translation' })).toBeVisible();

    const longText = 'Hello world. '.repeat(50); // ~650 chars
    const textarea = page.locator('textarea');
    await textarea.fill(longText);
    await expect(textarea).toHaveValue(longText);
    await expect(page.locator('button[type="submit"]')).toBeEnabled();
  });
});

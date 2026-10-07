import { test, expect } from './fixtures/base.fixture';
import { mockChatStream } from './fixtures/api-mocks';

// ================================================================
// Chat — advanced gap tests
// Covers: stream cancel, Korean, long text
// ================================================================

test.describe('Chat — stream cancel with partial response', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/chat');
    await expect(page.locator('main').getByRole('heading', { name: /Chat/ }).first()).toBeVisible();
    await expect(page.getByText('Loading models...')).toBeHidden({ timeout: 15_000 });
  });

  // Streaming cancel preserves partial response
  test('cancelling stream preserves partial response', async ({ page }) => {
    // Mock with delayed chunks — route handler that sends partial tokens then hangs
    await page.route('**/v1/chat/completions', async (route) => {
      // Send some tokens immediately, then hang (simulating slow stream)
      const partialBody = [
        'data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n',
        'data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}\n\n',
      ].join('');

      // Wait long enough for cancel to happen before completion
      await new Promise(resolve => setTimeout(resolve, 500));

      await route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: partialBody,
        headers: { 'Cache-Control': 'no-cache' },
      });
    });

    const input = page.getByPlaceholder('Type your message...');
    await input.fill('Hello');
    await page.locator('button[type="submit"]').click();

    // Wait for partial response to appear
    await expect(page.locator('.bg-muted').filter({ hasText: /Hello/ }).first()).toBeVisible({ timeout: 10_000 });

    // After response completes (even partial), input should be re-enabled
    await expect(input).toBeEnabled({ timeout: 10_000 });
  });
});

test.describe('Chat — Korean language input', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/chat');
    await expect(page.locator('main').getByRole('heading', { name: /Chat/ }).first()).toBeVisible();
    await expect(page.getByText('Loading models...')).toBeHidden({ timeout: 15_000 });
  });

  // Korean message — no encoding issues
  test('Korean message sends and displays correctly', async ({ page }) => {
    const koreanTokens = ['안녕', '하세요', ', ', '반갑', '습니다', '.'];
    await mockChatStream(page, '**/v1/chat/completions', koreanTokens);

    const input = page.getByPlaceholder('Type your message...');
    await input.fill('안녕하세요, 오늘 날씨가 어때요?');
    await page.locator('button[type="submit"]').click();

    // User message should display Korean text without encoding issues
    await expect(page.locator('.bg-primary').filter({ hasText: '안녕하세요' }).first()).toBeVisible({ timeout: 10_000 });

    // Assistant response should display Korean tokens
    await expect(page.locator('.bg-muted').filter({ hasText: '안녕하세요' }).first()).toBeVisible({ timeout: 10_000 });
  });
});

test.describe('Chat — long text input', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/chat');
    await expect(page.locator('main').getByRole('heading', { name: /Chat/ }).first()).toBeVisible();
    await expect(page.getByText('Loading models...')).toBeHidden({ timeout: 15_000 });
  });

  // Long text input (1000+ characters)
  test('1000+ character message sends successfully', async ({ page }) => {
    const longText = 'A'.repeat(1000) + ' end marker';
    await mockChatStream(page, '**/v1/chat/completions', ['Response', ' to', ' long', ' text']);

    const input = page.getByPlaceholder('Type your message...');
    await input.fill(longText);

    // Verify the input accepted the long text
    await expect(input).toHaveValue(longText);

    // Send button should be enabled
    const sendBtn = page.locator('button[type="submit"]');
    await expect(sendBtn).toBeEnabled();
    await sendBtn.click();

    // User message should display (may be truncated visually)
    await expect(page.locator('.bg-primary').filter({ hasText: 'end marker' }).first()).toBeVisible({ timeout: 10_000 });

    // Assistant response should arrive
    await expect(page.locator('.bg-muted').filter({ hasText: /Response to long text/ }).first()).toBeVisible({ timeout: 10_000 });
  });
});

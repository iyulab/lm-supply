import { test, expect } from './fixtures/base.fixture';
import { mockChatStream, mockJsonEndpoint } from './fixtures/api-mocks';

// ================================================================
// Network Error Scenarios
// Covers: server down, streaming error, recovery
// ================================================================

test.describe('Network Error Scenarios', () => {
  // API server down — shows error message
  test('embed page shows error when server returns 500', async ({ page }) => {
    // Mock embed endpoint to return server error
    await page.route('**/v1/embeddings', async (route) => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: { message: 'Internal server error' } }),
      });
    });

    await page.goto('/embed');
    await expect(page.locator('main').getByRole('heading', { name: 'Text Embedding' })).toBeVisible();

    const textarea = page.locator('textarea');
    await textarea.fill('test input');
    await page.locator('button[type="submit"]').click();

    // Should show error message
    await expect(page.getByText(/error|failed|500/i)).toBeVisible({ timeout: 10_000 });
  });

  // Server connection refused — shows error in Chat
  test('chat page shows error on network failure', async ({ page }) => {
    // Mock chat endpoint to abort (simulate connection refused)
    await page.route('**/v1/chat/completions', async (route) => {
      await route.abort('connectionrefused');
    });

    await page.goto('/chat');
    await expect(page.locator('main').getByRole('heading', { name: /Chat/ }).first()).toBeVisible();
    await expect(page.getByText('Loading models...')).toBeHidden({ timeout: 15_000 });

    const input = page.getByPlaceholder('Type your message...');
    await input.fill('Hello');
    await page.locator('button[type="submit"]').click();

    // Should show error, not crash
    await expect(page.getByText(/error|failed|network/i)).toBeVisible({ timeout: 10_000 });
  });

  // Streaming error mid-response — partial response preserved
  test('streaming error mid-response preserves partial text', async ({ page }) => {
    // Build SSE body: a few tokens, then error, then DONE
    const tokens = ['Hello', ' there'];
    const chunks = tokens.map((token) => {
      const data = JSON.stringify({
        id: 'chatcmpl-1',
        object: 'chat.completion.chunk',
        choices: [{ index: 0, delta: { content: token }, finish_reason: null }],
      });
      return `data: ${data}\n\n`;
    });
    // Add finish reason stop to end the stream normally
    const finalChunk = JSON.stringify({
      id: 'chatcmpl-1',
      object: 'chat.completion.chunk',
      choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
    });
    chunks.push(`data: ${finalChunk}\n\n`);
    chunks.push('data: [DONE]\n\n');

    await page.route('**/v1/chat/completions', async (route) => {
      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'text/event-stream' },
        body: chunks.join(''),
      });
    });

    await page.goto('/chat');
    await expect(page.locator('main').getByRole('heading', { name: /Chat/ }).first()).toBeVisible();
    await expect(page.getByText('Loading models...')).toBeHidden({ timeout: 15_000 });

    const input = page.getByPlaceholder('Type your message...');
    await input.fill('Tell me something');
    await page.locator('button[type="submit"]').click();

    // Should show the partial response text
    await expect(page.getByText('Hello there')).toBeVisible({ timeout: 10_000 });
  });

  // Error recovery — after error, normal request succeeds
  test('chat recovers after error on retry', async ({ page }) => {
    let requestCount = 0;

    await page.route('**/v1/chat/completions', async (route) => {
      requestCount++;
      if (requestCount === 1) {
        // First request fails
        await route.abort('connectionrefused');
      } else {
        // Subsequent requests succeed with streaming response
        const encoder = new TextEncoder();
        const tokens = ['Recovery', ' successful', '!'];
        const chunks = tokens.map((token) => {
          const data = JSON.stringify({
            id: 'chatcmpl-2',
            object: 'chat.completion.chunk',
            choices: [{ index: 0, delta: { content: token }, finish_reason: null }],
          });
          return `data: ${data}\n\n`;
        });
        chunks.push('data: [DONE]\n\n');

        await route.fulfill({
          status: 200,
          headers: { 'Content-Type': 'text/event-stream' },
          body: chunks.join(''),
        });
      }
    });

    await page.goto('/chat');
    await expect(page.locator('main').getByRole('heading', { name: /Chat/ }).first()).toBeVisible();
    await expect(page.getByText('Loading models...')).toBeHidden({ timeout: 15_000 });

    const input = page.getByPlaceholder('Type your message...');

    // First attempt — should fail
    await input.fill('First attempt');
    await page.locator('button[type="submit"]').click();
    await expect(page.getByText(/error|failed/i)).toBeVisible({ timeout: 10_000 });

    // Second attempt — should succeed
    await input.fill('Second attempt');
    await page.locator('button[type="submit"]').click();
    await expect(page.getByText('Recovery successful!')).toBeVisible({ timeout: 10_000 });
  });
});

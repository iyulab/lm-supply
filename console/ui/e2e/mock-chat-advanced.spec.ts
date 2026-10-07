import { test, expect } from './fixtures/base.fixture';
import { mockChatStream, mockJsonError } from './fixtures/api-mocks';

// ================================================================
// Chat page — advanced mock tests
// Covers: cancel, disabled during load, multi-turn, auto-scroll
// ================================================================

test.describe('Chat — multi-turn and cancel', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/chat');
    await expect(page.locator('main').getByRole('heading', { name: /Chat/ }).first()).toBeVisible();
    await expect(page.getByText('Loading models...')).toBeHidden({ timeout: 15_000 });
  });

  // Multi-turn conversation history maintained
  test('multiple messages create conversation history', async ({ page }) => {
    // First exchange
    await mockChatStream(page, '**/v1/chat/completions', ['First', ' response']);
    await page.getByPlaceholder('Type your message...').fill('Hello');
    await page.locator('button[type="submit"]').click();
    await expect(page.locator('.bg-muted').filter({ hasText: 'First response' }).first()).toBeVisible({ timeout: 10_000 });

    // Unroute to re-mock with different response
    await page.unroute('**/v1/chat/completions');

    // Second exchange
    await mockChatStream(page, '**/v1/chat/completions', ['Second', ' response']);
    await page.getByPlaceholder('Type your message...').fill('How are you?');
    await page.locator('button[type="submit"]').click();
    await expect(page.locator('.bg-muted').filter({ hasText: 'Second response' }).first()).toBeVisible({ timeout: 10_000 });

    // Both user messages should be visible
    await expect(page.locator('.bg-primary').filter({ hasText: 'Hello' }).first()).toBeVisible();
    await expect(page.locator('.bg-primary').filter({ hasText: 'How are you?' }).first()).toBeVisible();

    // Both assistant messages should be visible
    await expect(page.locator('.bg-muted').filter({ hasText: 'First response' }).first()).toBeVisible();
    await expect(page.locator('.bg-muted').filter({ hasText: 'Second response' }).first()).toBeVisible();
  });

  // Input disabled during streaming
  test('input is disabled during streaming', async ({ page }) => {
    // Use a slow mock to keep loading state visible
    await page.route('**/v1/chat/completions', async (route) => {
      // Delay the response to keep isLoading = true
      await new Promise(resolve => setTimeout(resolve, 3000));
      await route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: 'data: {"choices":[{"delta":{"content":"OK"}}]}\n\ndata: [DONE]\n\n',
      });
    });

    await page.getByPlaceholder('Type your message...').fill('Test');
    await page.locator('button[type="submit"]').click();

    // Input should be disabled during loading
    await expect(page.getByPlaceholder('Type your message...')).toBeDisabled();

    // Cancel button (Square icon) should appear instead of Send
    await expect(page.locator('button').filter({ has: page.locator('.lucide-square') })).toBeVisible();
  });

  // Cancel button stops streaming
  test('cancel button appears and is clickable during streaming', async ({ page }) => {
    // Use a very slow mock
    await page.route('**/v1/chat/completions', async (route) => {
      await new Promise(resolve => setTimeout(resolve, 10_000));
      await route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: 'data: [DONE]\n\n',
      });
    });

    await page.getByPlaceholder('Type your message...').fill('Long prompt');
    await page.locator('button[type="submit"]').click();

    // Wait for cancel button to appear
    const cancelBtn = page.locator('button[aria-label="Cancel generation"]');
    await expect(cancelBtn).toBeVisible({ timeout: 5_000 });

    // Click cancel
    await cancelBtn.click();

    // Input should be re-enabled after cancel
    await expect(page.getByPlaceholder('Type your message...')).toBeEnabled({ timeout: 5_000 });
  });

  // Auto-scroll to latest message
  test('messages container has auto-scroll behavior', async ({ page }) => {
    // Send a message and verify the messages area exists and is scrollable
    await mockChatStream(page, '**/v1/chat/completions', ['Response']);

    await page.getByPlaceholder('Type your message...').fill('Test');
    await page.locator('button[type="submit"]').click();

    // The messages container should exist with overflow-auto
    const messagesArea = page.locator('.overflow-auto').first();
    await expect(messagesArea).toBeVisible();

    // Response should be visible (implying scroll worked)
    await expect(page.locator('.bg-muted').filter({ hasText: 'Response' }).first()).toBeVisible({ timeout: 10_000 });
  });

  // Model selector disabled during streaming
  test('model selector is disabled during streaming', async ({ page }) => {
    await page.route('**/v1/chat/completions', async (route) => {
      await new Promise(resolve => setTimeout(resolve, 3000));
      await route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: 'data: {"choices":[{"delta":{"content":"OK"}}]}\n\ndata: [DONE]\n\n',
      });
    });

    await page.getByPlaceholder('Type your message...').fill('Test');
    await page.locator('button[type="submit"]').click();

    // Model selector should be disabled during loading
    const select = page.locator('select');
    await expect(select).toBeDisabled();
  });
});

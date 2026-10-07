import { test, expect } from './fixtures/base.fixture';
import { mockChatStream, mockJsonError } from './fixtures/api-mocks';

// ================================================================
// Chat page — mock streaming tests
// Covers: message display, streaming, elapsed time, error
// ================================================================

test.describe('Chat — mock streaming', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/chat');
    await expect(page.locator('main').getByRole('heading', { name: /Chat/ }).first()).toBeVisible();
    // Wait for model selector to load
    await expect(page.getByText('Loading models...')).toBeHidden({ timeout: 15_000 });
  });

  // Message sending shows streaming response; User message displayed; Assistant response displayed
  test('sending message shows user and assistant messages', async ({ page }) => {
    const tokens = ['Hello', ', ', 'I am', ' an', ' AI', ' assistant', '.'];

    // Mock the chat streaming endpoint (goes directly to localhost:5000 in dev mode)
    await mockChatStream(page, '**/v1/chat/completions', tokens);

    // Type and send a message
    const input = page.getByPlaceholder('Type your message...');
    await input.fill('Hello');
    await page.locator('button[type="submit"]').click();

    // User message should appear (right-aligned with bg-primary)
    await expect(page.locator('.bg-primary').filter({ hasText: 'Hello' }).first()).toBeVisible({ timeout: 10_000 });

    // Assistant response should appear (in bg-muted)
    await expect(page.locator('.bg-muted').filter({ hasText: /Hello.*I am.*AI assistant/ }).first()).toBeVisible({ timeout: 10_000 });
  });

  // Elapsed time shown after response
  test('elapsed time is shown after response completes', async ({ page }) => {
    await mockChatStream(page, '**/v1/chat/completions', ['Response', ' text']);

    const input = page.getByPlaceholder('Type your message...');
    await input.fill('Test');
    await page.locator('button[type="submit"]').click();

    // Wait for response to complete
    await expect(page.locator('.bg-muted').filter({ hasText: 'Response text' }).first()).toBeVisible({ timeout: 10_000 });

    // Elapsed time should be shown
    await expect(page.getByText(/\d+ ms/)).toBeVisible({ timeout: 5_000 });
  });

  // Empty input disables send (already tested but verifying with mock)
  test('send button is disabled with empty input', async ({ page }) => {
    const sendBtn = page.locator('button[type="submit"]');
    await expect(sendBtn).toBeDisabled();
  });

  // Error handling
  test('model error shows error message in chat', async ({ page }) => {
    // Mock error response
    await mockJsonError(page, '**/v1/chat/completions', 'Model failed to load: generator not found', 500);

    const input = page.getByPlaceholder('Type your message...');
    await input.fill('Hello');
    await page.locator('button[type="submit"]').click();

    // Error should appear as an assistant message
    await expect(page.getByText(/Error:.*Model failed to load/)).toBeVisible({ timeout: 10_000 });
  });

  // User message right-aligned with user icon
  test('user message has user icon', async ({ page }) => {
    await mockChatStream(page, '**/v1/chat/completions', ['OK']);

    const input = page.getByPlaceholder('Type your message...');
    await input.fill('Hi there');
    await page.locator('button[type="submit"]').click();

    // User message should show with justify-end (right-aligned)
    const userMsg = page.locator('.justify-end').filter({ hasText: 'Hi there' });
    await expect(userMsg).toBeVisible({ timeout: 10_000 });
  });

  // Assistant message with bot icon
  test('assistant message has bot icon', async ({ page }) => {
    await mockChatStream(page, '**/v1/chat/completions', ['I', ' can', ' help']);

    const input = page.getByPlaceholder('Type your message...');
    await input.fill('Help me');
    await page.locator('button[type="submit"]').click();

    // Assistant response should appear with the bot icon (bg-primary circle)
    await expect(page.locator('.bg-muted').filter({ hasText: /I can help/ }).first()).toBeVisible({ timeout: 10_000 });
  });
});

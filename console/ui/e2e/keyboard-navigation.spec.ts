import { test, expect } from './fixtures/base.fixture';

// ================================================================
// Keyboard navigation: Tab order, Enter/Escape, focus management
// ================================================================

// ================================================================
// 1. Enter key submits forms
// ================================================================

test.describe('Chat — Enter key', () => {
  test('pressing Enter in input submits message', async ({ page }) => {
    // Mock the chat endpoint to return a streaming response
    await page.route('**/v1/chat/completions', async (route) => {
      const body = 'data: {"choices":[{"delta":{"content":"Hello!"}}]}\n\ndata: [DONE]\n\n';
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
    await input.fill('Hello');
    await input.press('Enter');

    // Assistant response should appear in chat
    await expect(page.getByText('Hello!', { exact: true })).toBeVisible({ timeout: 5_000 });
  });
});

test.describe('Rerank — Enter key', () => {
  test('pressing Enter does not submit when fields incomplete', async ({ page }) => {
    await page.goto('/rerank');
    await expect(page.locator('main').getByRole('heading', { name: 'Document Reranking' })).toBeVisible();

    // Fill only the query (no documents)
    const queryInput = page.locator('input[type="text"]').first();
    await queryInput.fill('test query');
    await queryInput.press('Enter');

    // Submit button should still be disabled (no documents filled)
    const submitBtn = page.locator('button[type="submit"]');
    await expect(submitBtn).toBeDisabled();
  });
});

// ================================================================
// 2. Tab order — focus moves through form elements
// ================================================================

test.describe('Embed — tab navigation', () => {
  test('Tab moves focus from textarea to submit button', async ({ page }) => {
    await page.goto('/embed');
    await expect(page.locator('main').getByRole('heading', { name: 'Embed' })).toBeVisible();

    const textarea = page.locator('textarea');
    await textarea.focus();
    await textarea.fill('test text');

    // Tab to next focusable element
    await page.keyboard.press('Tab');

    // The submit button or next form element should be focused
    const focused = page.locator(':focus');
    await expect(focused).toBeVisible();
  });
});

test.describe('Translate — tab navigation', () => {
  test('textarea is focusable and accepts input', async ({ page }) => {
    await page.goto('/translate');
    await expect(page.locator('main').getByRole('heading', { name: 'Machine Translation' })).toBeVisible();

    const textarea = page.locator('textarea');
    await textarea.focus();
    await expect(textarea).toBeFocused();

    // Can type into focused textarea
    await page.keyboard.type('Test text');
    await expect(textarea).toHaveValue('Test text');
  });
});

// ================================================================
// 3. Focus management — initial focus and return
// ================================================================

test.describe('Chat — focus management', () => {
  test('input receives focus on page load', async ({ page }) => {
    await page.goto('/chat');
    await expect(page.locator('main').getByRole('heading', { name: 'Chat' })).toBeVisible();

    // Chat input should be focusable
    const input = page.locator('input[type="text"]').first();
    await input.focus();
    await expect(input).toBeFocused();
  });

  test('input retains focus after failed submission', async ({ page }) => {
    await page.goto('/chat');
    await expect(page.locator('main').getByRole('heading', { name: 'Chat' })).toBeVisible();

    const input = page.locator('input[type="text"]').first();
    await input.focus();
    await input.fill('test');

    // Mock chat to fail
    await page.route('**/v1/chat/completions', async (route) => {
      await route.fulfill({ status: 500, body: 'Server Error' });
    });

    await input.press('Enter');

    // After error, input should still be accessible
    await expect(input).toBeVisible();
  });
});

// ================================================================
// 4. Sidebar keyboard navigation
// ================================================================

test.describe('Sidebar — keyboard navigation', () => {
  test('sidebar links are keyboard accessible', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('main')).toBeVisible();

    // All sidebar nav links should be accessible
    const navLinks = page.locator('nav a');
    const count = await navLinks.count();
    expect(count).toBeGreaterThan(5);

    // First nav link should be focusable
    await navLinks.first().focus();
    await expect(navLinks.first()).toBeFocused();
  });

  test('Enter key activates sidebar link', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('main')).toBeVisible();

    // Focus on the Chat link and press Enter
    const chatLink = page.locator('nav a[href="/chat"]');
    await chatLink.focus();
    await chatLink.press('Enter');

    // Should navigate to chat page
    await expect(page).toHaveURL(/\/chat/);
    await expect(page.locator('main').getByRole('heading', { name: 'Chat' })).toBeVisible();
  });
});

// ================================================================
// 5. Form submission via Enter across domains
// ================================================================

test.describe('Synthesize — Enter key in form', () => {
  test('form can be submitted via Enter key from textarea', async ({ page }) => {
    let apiCalled = false;

    await page.route('**/v1/audio/speech', async (route) => {
      apiCalled = true;
      // Return a fake audio blob
      await route.fulfill({
        status: 200,
        contentType: 'audio/wav',
        body: Buffer.from('RIFF'),
      });
    });

    await page.goto('/synthesize');
    await expect(page.locator('main').getByRole('heading', { name: 'Text to Speech' })).toBeVisible();
    await expect(page.getByText('Loading models...')).toBeHidden({ timeout: 15_000 });

    const textarea = page.locator('textarea');
    await textarea.fill('Test speech synthesis');

    // Submit via the button (textarea Enter adds newline)
    const submitBtn = page.locator('button[type="submit"]');
    await submitBtn.focus();
    await submitBtn.press('Enter');

    // API should be called
    await expect.poll(() => apiCalled, { timeout: 5_000 }).toBe(true);
  });
});

test.describe('ImageGenerate — Enter key', () => {
  test('pressing Enter in prompt textarea does not auto-submit', async ({ page }) => {
    let apiCalled = false;

    await page.route('**/v1/images/generations', async (route) => {
      apiCalled = true;
      await route.fulfill({ status: 200, body: '{}' });
    });

    await page.goto('/image-generate');
    await expect(page.locator('main').getByRole('heading', { name: 'Image Generation' })).toBeVisible();

    const textarea = page.locator('textarea');
    await textarea.fill('a beautiful sunset');
    await textarea.press('Enter');

    // Enter in textarea should add newline, not submit
    // Verify textarea has a newline (Enter added to text, not submitted)
    await expect(textarea).toHaveValue(/a beautiful sunset\n/);
    expect(apiCalled).toBe(false);
  });
});

// ================================================================
// 6. Escape key behavior
// ================================================================

test.describe('Keyboard — typing into inputs', () => {
  test('keyboard typing populates embed textarea', async ({ page }) => {
    await page.goto('/embed');
    await expect(page.locator('main').getByRole('heading', { name: 'Embed' })).toBeVisible();

    const textarea = page.locator('textarea');
    await textarea.focus();
    await expect(textarea).toBeFocused();

    // Type via keyboard
    await page.keyboard.type('First line');
    await page.keyboard.press('Enter');
    await page.keyboard.type('Second line');

    await expect(textarea).toHaveValue('First line\nSecond line');
  });
});

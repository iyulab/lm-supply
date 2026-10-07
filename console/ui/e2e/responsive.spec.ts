import { test, expect } from './fixtures/base.fixture';

// ================================================================
// Mobile responsive testing: viewport behavior at various widths
// ================================================================

// ================================================================
// 1. Desktop viewport (1280px) — baseline
// ================================================================

test.describe('Desktop viewport (1280px)', () => {
  test.beforeEach(async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 720 });
  });

  test('sidebar is visible at desktop width', async ({ page }) => {
    await page.goto('/');
    const sidebar = page.locator('aside');
    await expect(sidebar).toBeVisible();
  });

  test('main content area is visible alongside sidebar', async ({ page }) => {
    await page.goto('/chat');
    await expect(page.locator('main')).toBeVisible();
    await expect(page.locator('aside')).toBeVisible();
  });

  test('dashboard uses multi-column grid at desktop', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('main')).toBeVisible();
    // Dashboard grid should render (cards exist)
    const cards = page.locator('.grid > div');
    await expect(cards.first()).toBeVisible();
  });
});

// ================================================================
// 2. Tablet viewport (1024px) — lg breakpoint
// ================================================================

test.describe('Tablet viewport (1024px)', () => {
  test.beforeEach(async ({ page }) => {
    await page.setViewportSize({ width: 1024, height: 768 });
  });

  test('sidebar remains visible at tablet width', async ({ page }) => {
    await page.goto('/');
    const sidebar = page.locator('aside');
    await expect(sidebar).toBeVisible();
  });

  test('main content is accessible at tablet width', async ({ page }) => {
    await page.goto('/embed');
    await expect(page.locator('main').getByRole('heading', { name: /Embed|Text Embedding/ })).toBeVisible();
    // Textarea should be usable
    const textarea = page.locator('textarea');
    await expect(textarea).toBeVisible();
    await textarea.fill('test');
    await expect(textarea).toHaveValue('test');
  });

  test('ImageGenerate layout at tablet width', async ({ page }) => {
    await page.goto('/image-generate');
    await expect(page.locator('main').getByRole('heading', { name: 'Image Generation' })).toBeVisible();
    // Form and preview should both be visible
    const textarea = page.locator('textarea');
    await expect(textarea).toBeVisible();
  });
});

// ================================================================
// 3. Mobile viewport (768px) — md breakpoint
// ================================================================

test.describe('Mobile viewport (768px)', () => {
  test.beforeEach(async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 1024 });
  });

  test('sidebar is visible at mobile width', async ({ page }) => {
    await page.goto('/');
    const sidebar = page.locator('aside');
    await expect(sidebar).toBeVisible();
  });

  test('chat page is functional at mobile width', async ({ page }) => {
    await page.goto('/chat');
    await expect(page.locator('main').getByRole('heading', { name: 'Chat' })).toBeVisible();

    const input = page.locator('input[type="text"]').first();
    await expect(input).toBeVisible();
    await input.fill('Hello mobile');
    await expect(input).toHaveValue('Hello mobile');
  });

  test('translate form adapts at mobile width', async ({ page }) => {
    await page.goto('/translate');
    await expect(page.locator('main').getByRole('heading', { name: 'Machine Translation' })).toBeVisible();

    const textarea = page.locator('textarea');
    await expect(textarea).toBeVisible();
    await textarea.fill('Mobile translation test');
    await expect(textarea).toHaveValue('Mobile translation test');
  });

  test('models page is functional at mobile width', async ({ page }) => {
    await page.goto('/models');
    await expect(page.locator('main').getByRole('heading', { name: 'Model Management' })).toBeVisible();
  });
});

// ================================================================
// 4. Narrow mobile viewport (480px) — below all breakpoints
// ================================================================

test.describe('Narrow mobile viewport (480px)', () => {
  test.beforeEach(async ({ page }) => {
    await page.setViewportSize({ width: 480, height: 800 });
  });

  test('page loads without errors at narrow width', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('main')).toBeVisible();
  });

  test('embed page form elements are visible at narrow width', async ({ page }) => {
    await page.goto('/embed');
    // Main heading should be visible (may need scrolling)
    const heading = page.locator('main').getByRole('heading').first();
    await expect(heading).toBeVisible();
  });

  test('navigation links remain accessible at narrow width', async ({ page }) => {
    await page.goto('/');
    const navLinks = page.locator('nav a');
    const count = await navLinks.count();
    expect(count).toBeGreaterThan(5);
  });
});

// ================================================================
// 5. Viewport resize behavior
// ================================================================

test.describe('Viewport resize', () => {
  test('page handles viewport resize from desktop to mobile', async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 720 });
    await page.goto('/embed');
    await expect(page.locator('main').getByRole('heading', { name: /Embed|Text Embedding/ })).toBeVisible();

    // Resize to mobile
    await page.setViewportSize({ width: 480, height: 800 });

    // Content should still be accessible
    const textarea = page.locator('textarea');
    await expect(textarea).toBeVisible();
  });

  test('page handles viewport resize from mobile to desktop', async ({ page }) => {
    await page.setViewportSize({ width: 480, height: 800 });
    await page.goto('/chat');
    await expect(page.locator('main').getByRole('heading', { name: 'Chat' })).toBeVisible();

    // Resize to desktop
    await page.setViewportSize({ width: 1280, height: 720 });

    // Sidebar and content should both be visible
    await expect(page.locator('aside')).toBeVisible();
    await expect(page.locator('main')).toBeVisible();
  });
});

// ================================================================
// 6. Content overflow checks
// ================================================================

test.describe('Content overflow', () => {
  test('long text in embed textarea does not overflow container', async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.goto('/embed');
    await expect(page.locator('main').getByRole('heading', { name: /Embed|Text Embedding/ })).toBeVisible();

    const textarea = page.locator('textarea');
    const longText = 'A'.repeat(500) + '\n' + 'B'.repeat(500);
    await textarea.fill(longText);

    // Textarea should contain the text (not be truncated)
    const value = await textarea.inputValue();
    expect(value.length).toBe(longText.length);
  });

  test('sidebar navigation does not overlap main content', async ({ page }) => {
    await page.setViewportSize({ width: 1024, height: 768 });
    await page.goto('/chat');

    const sidebar = page.locator('aside');
    const main = page.locator('main');

    const sidebarBox = await sidebar.boundingBox();
    const mainBox = await main.boundingBox();

    if (sidebarBox && mainBox) {
      // Sidebar right edge should not overlap main left edge
      expect(sidebarBox.x + sidebarBox.width).toBeLessThanOrEqual(mainBox.x + 1);
    }
  });
});

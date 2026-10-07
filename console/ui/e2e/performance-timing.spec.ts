import { test, expect } from './fixtures/base.fixture';

// ================================================================
// Performance & Load Timing
// Covers: initial load, page switch
// ================================================================

test.describe('Performance — load timing', () => {
  // Initial dashboard load within 5 seconds
  test('dashboard loads within 5 seconds', async ({ page }) => {
    const start = Date.now();
    await page.goto('/');
    await expect(page.getByRole('heading', { name: /Dashboard/ })).toBeVisible();
    const elapsed = Date.now() - start;

    // Should load within 5 seconds
    expect(elapsed).toBeLessThan(5000);
  });

  // Page switch within 1 second
  test('page switch from dashboard to embed is instant', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByRole('heading', { name: /Dashboard/ })).toBeVisible();

    // Click Embed link and measure
    const start = Date.now();
    await page.getByRole('link', { name: /Embed/ }).click();
    await expect(page.locator('main').getByRole('heading', { name: 'Text Embedding' })).toBeVisible();
    const elapsed = Date.now() - start;

    expect(elapsed).toBeLessThan(1000);
  });

  // Page switch between multiple pages
  test('rapid page switches all under 1 second', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByRole('heading', { name: /Dashboard/ })).toBeVisible();

    const pages = [
      { link: /Chat/, heading: /Chat/ },
      { link: /Embed/, heading: 'Text Embedding' },
      { link: /Translate/, heading: 'Machine Translation' },
      { link: /Detect/, heading: 'Object Detection' },
    ];

    for (const p of pages) {
      const start = Date.now();
      await page.getByRole('link', { name: p.link }).click();
      await expect(page.locator('main').getByRole('heading', { name: p.heading }).first()).toBeVisible();
      const elapsed = Date.now() - start;
      expect(elapsed).toBeLessThan(1000);
    }
  });
});

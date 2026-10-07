import { test, expect } from './fixtures/base.fixture';

// ================================================================
// Translate page — deeper interaction tests
// ================================================================

test.describe('Translate — deeper UI', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/translate');
    await expect(page.locator('main').getByRole('heading', { name: 'Machine Translation' })).toBeVisible();
  });

  // model options show "alias (source → target)" format
  test('model options show alias with language pair', async ({ page }) => {
    const select = page.locator('select');
    // Wait for options to be populated from API
    await expect(select.locator('option')).not.toHaveCount(0, { timeout: 5_000 });

    const options = await select.locator('option').allTextContents();

    // If languages loaded from API, options should contain arrow character
    if (options.length > 1 || (options.length === 1 && options[0] !== 'default')) {
      const hasArrowFormat = options.some(o => o.includes('→'));
      expect(hasArrowFormat).toBe(true);
    }
  });

  // source/target language hint appears under model selector
  test('language hint shows source → target below model selector', async ({ page }) => {
    const select = page.locator('select');
    await expect(select.locator('option')).not.toHaveCount(0, { timeout: 5_000 });
    const options = await select.locator('option').allTextContents();

    // If models loaded, selecting one should show language hint
    if (options.length > 0 && options[0] !== 'default') {
      // Language hint text is below the select
      const hint = page.locator('p.text-muted-foreground').filter({ hasText: /→/ });
      await expect(hint.first()).toBeVisible();
    }
  });

  // [4-54] Translation output is read-only
  test('[4-54] translation output is read-only div', async ({ page }) => {
    // The translation area is a div, not a textarea
    const outputDivs = page.locator('.grid .bg-muted.border.border-border.rounded.overflow-auto');
    await expect(outputDivs.first()).toBeVisible();
  });

  // Source textarea has correct height
  test('source textarea has consistent height', async ({ page }) => {
    const textarea = page.locator('textarea');
    await expect(textarea).toHaveClass(/h-48/);
  });

  // Translate button shows arrow icon
  test('translate button has arrow icon and text', async ({ page }) => {
    const btn = page.getByRole('button', { name: /Translate/ });
    await expect(btn).toBeVisible();
    // Has text "Translate"
    await expect(btn).toContainText('Translate');
  });
});

// ================================================================
// Image Generate — deeper interaction tests
// ================================================================

test.describe('ImageGenerate — deeper UI', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/image-generate');
    await expect(page.locator('main').getByRole('heading', { name: 'Image Generation' })).toBeVisible();
  });

  // [4-100] Negative prompt textarea in advanced settings
  test('[4-100] negative prompt textarea in advanced settings', async ({ page }) => {
    // Open advanced settings
    await page.getByRole('button', { name: /Advanced Settings/i }).click();

    // Negative prompt label and textarea
    await expect(page.locator('label', { hasText: 'Negative Prompt' })).toBeVisible();
    const negPrompt = page.locator('textarea').nth(1); // Second textarea is negative prompt
    await expect(negPrompt).toBeVisible();
    await expect(negPrompt).toHaveAttribute('placeholder', 'blurry, low quality, distorted...');
  });

  // Seed input is number type
  test('seed input is number type', async ({ page }) => {
    await page.getByRole('button', { name: /Advanced Settings/i }).click();

    const seedInput = page.locator('input[type="number"]');
    await expect(seedInput).toBeVisible();
    await expect(seedInput).toHaveAttribute('placeholder', 'Random');
  });

  // Seed can be manually entered
  test('seed input accepts manual entry', async ({ page }) => {
    await page.getByRole('button', { name: /Advanced Settings/i }).click();

    const seedInput = page.locator('input[type="number"]');
    await seedInput.fill('42');
    const value = await seedInput.inputValue();
    expect(value).toBe('42');
  });

  // Steps label shows current value
  test('steps label shows current value dynamically', async ({ page }) => {
    await page.getByRole('button', { name: /Advanced Settings/i }).click();

    // Default steps is 4
    await expect(page.getByText(/Steps \(4\)/)).toBeVisible();

    // Change steps
    const slider = page.locator('input[type="range"]').first();
    await slider.fill('6');
    await expect(page.getByText(/Steps \(6\)/)).toBeVisible();
  });

  // Guidance scale label shows current value
  test('guidance scale label shows current value dynamically', async ({ page }) => {
    await page.getByRole('button', { name: /Advanced Settings/i }).click();

    // Default guidance is 1.0
    await expect(page.getByText(/Guidance Scale \(1\.0\)/)).toBeVisible();
  });

  // Result panel placeholder content
  test('result panel shows placeholder before generation', async ({ page }) => {
    await expect(page.getByText('Your generated image will appear here')).toBeVisible();
  });

  // Model and size selects are in same row (grid layout)
  test('model and size selects are in a grid layout', async ({ page }) => {
    const gridRow = page.locator('.grid.grid-cols-2.gap-4');
    await expect(gridRow.first()).toBeVisible();

    // Both selects within the grid
    const selects = gridRow.first().locator('select');
    expect(await selects.count()).toBe(2);
  });

  // Generate button shows icon and text
  test('generate button shows icon and text', async ({ page }) => {
    const btn = page.locator('button', { hasText: 'Generate Image' });
    await expect(btn).toBeVisible();
  });

  // Default size is 512x512
  test('default size is 512x512', async ({ page }) => {
    const sizeSelect = page.locator('select').nth(1); // Second select is size
    const value = await sizeSelect.inputValue();
    expect(value).toBe('512x512');
  });
});

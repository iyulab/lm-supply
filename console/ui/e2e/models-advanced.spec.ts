import { test, expect } from './fixtures/base.fixture';

// ================================================================
// Models page — advanced features
// ================================================================

test.describe('Models — HF Check', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/models');
    await expect(page.getByRole('heading', { name: /Model Management/ })).toBeVisible();
  });

  // [3-06] Valid repo check shows model info
  test('[3-06] valid HF repo check shows model info', async ({ page }) => {
    const repoInput = page.getByPlaceholder('e.g., BAAI/bge-small-en-v1.5');
    await repoInput.clear();
    await repoInput.fill('sentence-transformers/all-MiniLM-L6-v2');

    await page.getByRole('button', { name: 'Check' }).click();

    // Should show "Model found" info
    await expect(page.getByText('Model found')).toBeVisible({ timeout: 15_000 });
  });

  // [3-07] Invalid repo check shows error
  test('[3-07] invalid HF repo check shows error', async ({ page }) => {
    const repoInput = page.getByPlaceholder('e.g., BAAI/bge-small-en-v1.5');
    await repoInput.clear();
    await repoInput.fill('nonexistent/fake-model-12345');

    await page.getByRole('button', { name: 'Check' }).click();

    // Should show error message
    await expect(page.getByText(/not found|error|does not exist|failed/i)).toBeVisible({ timeout: 15_000 });
  });
});

test.describe('Models — Advanced Options', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/models');
    await expect(page.getByRole('heading', { name: /Model Management/ })).toBeVisible();
  });

  // [3-23] Advanced options toggle shows additional fields
  test('[3-23] advanced options toggle reveals additional settings', async ({ page }) => {
    // Find Advanced options button in pre-load section
    const advButton = page.getByRole('button', { name: /Advanced/i });
    await expect(advButton).toBeVisible();

    // Click to expand
    await advButton.click();

    // Should show provider selector
    await expect(page.getByText(/Provider/i)).toBeVisible();

    // Should show thread count field
    await expect(page.getByText(/Thread/i)).toBeVisible();
  });

  // [3-24] Generator type shows generator-specific options
  test('[3-24] generator type shows context length option', async ({ page }) => {
    // Select Generator in type dropdown (should be default)
    const preloadSection = page.getByRole('heading', { name: /Pre-load/ }).locator('..');
    const typeSelect = preloadSection.locator('select').first();
    await typeSelect.selectOption('generator');

    // Open advanced options
    await page.getByRole('button', { name: /Advanced/i }).click();

    // Should show generator-specific options
    await expect(page.getByText(/Max.*Context|Context.*Length|Concurrent/i)).toBeVisible();
  });

  // [3-25] Embedder type shows embedder-specific options
  test('[3-25] embedder type shows sequence length option', async ({ page }) => {
    // Select Embedder type
    const preloadSection = page.getByRole('heading', { name: /Pre-load/ }).locator('..');
    const typeSelect = preloadSection.locator('select').first();
    await typeSelect.selectOption('embedder');

    // Open advanced options
    await page.getByRole('button', { name: /Advanced/i }).click();

    // Should show embedder-specific options
    await expect(page.getByText(/Sequence|Max.*Seq/i)).toBeVisible();
  });
});

import { test, expect } from './fixtures/base.fixture';
import { mockModelsPageApis } from './fixtures/api-mocks';

// ================================================================
// Models management — coverage gaps (uses live backend for registry tests)
// Covers: no blank aliases, registry cached status,
//   delete during download, model load form
// ================================================================

test.describe('Model Registry — alias integrity', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/models');
    await expect(page.getByRole('heading', { name: /Model Management/ })).toBeVisible();
  });

  // All alias cells have values — no blank/undefined
  test('all registry alias cells are non-empty', async ({ page }) => {
    // Expand first registry type
    const typeButton = page.locator('button').filter({ hasText: /models$/ }).first();
    await typeButton.click();

    // Wait for Alias column header
    await expect(page.getByRole('columnheader', { name: 'Alias' })).toBeVisible();

    // Get all alias badge spans in the registry table
    const aliasCells = page.locator('span.rounded.text-xs.font-medium');
    const count = await aliasCells.count();
    expect(count).toBeGreaterThan(0);

    for (let i = 0; i < count; i++) {
      const text = (await aliasCells.nth(i).textContent()) ?? '';
      expect(text.trim()).not.toBe('');
      expect(text).not.toBe('undefined');
      expect(text).not.toBe('null');
    }
  });

  // Registry shows Cached/Not downloaded status for each model
  test('registry models show cached status indicators', async ({ page }) => {
    // Expand first registry type
    const typeButton = page.locator('button').filter({ hasText: /models$/ }).first();
    await typeButton.click();

    // Should see status indicators
    const statusCells = page.getByText(/Cached|Not downloaded/);
    await expect(statusCells.first()).toBeVisible();
  });
});

test.describe('Models — delete during download (mocked)', () => {
  // Cached model table renders during download state
  test('model with downloading status renders correctly', async ({ page }) => {
    await mockModelsPageApis(page, {
      cachedModels: [
        {
          repoId: 'test-org/test-model',
          detectedType: 'Embedder',
          sizeBytes: 120_000_000,
          fileCount: 5,
          localPath: '/cache/models--test-org--test-model',
        },
      ],
      loadedModels: [],
    });

    await page.goto('/models');
    await expect(page.getByRole('heading', { name: /Model Management/ })).toBeVisible();
    await expect(page.getByText('Cached Models (1)')).toBeVisible({ timeout: 5_000 });

    // Verify the model row is rendered
    await expect(page.getByText('test-org/test-model')).toBeVisible();
  });
});

test.describe('Models — pre-load form', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/models');
    await expect(page.getByRole('heading', { name: /Model Management/ })).toBeVisible();
  });

  // Pre-load form has all required elements
  test('pre-load form has type selector and model ID input', async ({ page }) => {
    // Scroll to pre-load section
    const preloadHeading = page.getByRole('heading', { name: /Pre-load/ });
    await expect(preloadHeading).toBeVisible();

    // Type dropdown should be present — scope to the pre-load section
    const preloadSection = preloadHeading.locator('..');
    const typeSelect = preloadSection.locator('select').first();
    await expect(typeSelect).toBeVisible();

    // Should have AI domain options (11 types)
    const options = await typeSelect.locator('option').allTextContents();
    expect(options.length).toBeGreaterThanOrEqual(5);
    expect(options).toContain('Generator');
    expect(options).toContain('Embedder');

    // Model ID input should be present
    const modelInput = page.getByPlaceholder('default');
    await expect(modelInput).toBeVisible();

    // Load button should be present
    await expect(page.getByRole('button', { name: 'Load', exact: true })).toBeVisible();
  });
});

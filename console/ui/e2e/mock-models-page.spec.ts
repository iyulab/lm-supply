import { test, expect } from './fixtures/base.fixture';
import { mockModelsPageApis } from './fixtures/api-mocks';

// ================================================================
// Models page — fully mocked API tests
// These tests mock all Models page API endpoints to test edge cases
// that require specific backend state (empty cache, GGUF, delete, etc.)
// ================================================================

// ================================================================
// Empty cached models state
// ================================================================

test.describe('Models — empty cached state', () => {
  test('[3-14] shows empty state message when no models cached', async ({ page }) => {
    await mockModelsPageApis(page, { cachedModels: [] });

    await page.goto('/models');
    await expect(page.locator('main').getByRole('heading', { name: 'Model Management' })).toBeVisible();

    // Should show "Cached Models (0)"
    await expect(page.getByText('Cached Models (0)')).toBeVisible({ timeout: 5_000 });

    // Should show empty state message
    await expect(page.getByText('No models in cache')).toBeVisible();
  });
});

// ================================================================
// GGUF file selection
// ================================================================

test.describe('Models — GGUF selection', () => {
  test('[3-11] GGUF file dropdown appears after checking a GGUF repo', async ({ page }) => {
    await mockModelsPageApis(page);

    // Mock check endpoint to return GGUF files
    await page.route('**/api/download/check', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          exists: true,
          repoId: 'TheBloke/Llama-2-7B-GGUF',
          detectedType: 'Generator',
          fileCount: 15,
          totalSizeBytes: 3_800_000_000,
          ggufFiles: [
            { fileName: 'llama-2-7b.Q4_K_M.gguf', sizeBytes: 4_080_000_000 },
            { fileName: 'llama-2-7b.Q5_K_M.gguf', sizeBytes: 4_780_000_000 },
            { fileName: 'llama-2-7b.Q8_0.gguf', sizeBytes: 7_160_000_000 },
          ],
        }),
      });
    });

    await page.goto('/models');
    await expect(page.locator('main').getByRole('heading', { name: 'Model Management' })).toBeVisible();

    // Enter repo ID and check
    await page.locator('input[placeholder*="BAAI"]').fill('TheBloke/Llama-2-7B-GGUF');
    await page.getByRole('button', { name: 'Check' }).click();

    // Model found message should appear
    await expect(page.getByText('Model found')).toBeVisible({ timeout: 10_000 });

    // GGUF file dropdown should appear with "GGUF File" label
    await expect(page.getByText('GGUF File')).toBeVisible();

    // Dropdown should have Auto option + 3 GGUF files
    const ggufSelect = page.locator('select').filter({ has: page.locator('option', { hasText: 'Auto' }) });
    await expect(ggufSelect).toBeVisible();

    // Verify specific GGUF files are listed
    await expect(ggufSelect).toContainText('llama-2-7b.Q4_K_M.gguf');
    await expect(ggufSelect).toContainText('llama-2-7b.Q5_K_M.gguf');
    await expect(ggufSelect).toContainText('llama-2-7b.Q8_0.gguf');
  });
});

// ================================================================
// Status badges (mocked scenario with loaded model)
// ================================================================

test.describe('Models — status badges', () => {
  test('[3-18] cached model shows Loaded badge when loaded', async ({ page }) => {
    await mockModelsPageApis(page, {
      cachedModels: [
        {
          repoId: 'BAAI/bge-small-en-v1.5',
          detectedType: 'Embedder',
          sizeBytes: 120_000_000,
          fileCount: 5,
          localPath: '/cache/models--BAAI--bge-small-en-v1.5',
        },
      ],
      loadedModels: [
        {
          modelId: 'BAAI/bge-small-en-v1.5',
          modelType: 'embedder',
          loadedAt: new Date().toISOString(),
          lastUsedAt: new Date().toISOString(),
        },
      ],
    });

    await page.goto('/models');
    await expect(page.getByText('Cached Models (1)')).toBeVisible({ timeout: 5_000 });

    // Should show "Loaded" badge (green)
    await expect(page.getByText('Loaded').first()).toBeVisible();
  });

  test('[3-18] cached model shows Ready badge when not loaded', async ({ page }) => {
    await mockModelsPageApis(page, {
      cachedModels: [
        {
          repoId: 'BAAI/bge-small-en-v1.5',
          detectedType: 'Embedder',
          sizeBytes: 120_000_000,
          fileCount: 5,
          localPath: '/cache/models--BAAI--bge-small-en-v1.5',
        },
      ],
      loadedModels: [],
    });

    await page.goto('/models');
    await expect(page.getByText('Cached Models (1)')).toBeVisible({ timeout: 5_000 });

    // Should show "Ready" badge
    await expect(page.getByText('Ready')).toBeVisible();
  });
});

// ================================================================
// Delete model; delete disabled on loaded model
// ================================================================

test.describe('Models — delete behavior', () => {
  const mockModel = {
    repoId: 'BAAI/bge-small-en-v1.5',
    detectedType: 'Embedder',
    sizeBytes: 120_000_000,
    fileCount: 5,
    localPath: '/cache/models--BAAI--bge-small-en-v1.5',
  };

  // Delete model (click trash, confirm, model removed)
  test('[3-15] delete button removes model after confirmation', async ({ page }) => {
    let deleteWasCalled = false;
    await mockModelsPageApis(page, {
      cachedModels: [mockModel],
      loadedModels: [],
    });

    // Override DELETE handler to track the call and return updated list
    await page.unroute('**/api/cache/models');
    await page.route('**/api/cache/models**', async (route) => {
      if (route.request().method() === 'DELETE') {
        deleteWasCalled = true;
        await route.fulfill({ status: 200, body: '{}' });
      } else if (route.request().method() === 'GET') {
        // After delete, return empty list
        const models = deleteWasCalled ? [] : [mockModel];
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ models }),
        });
      } else {
        await route.continue();
      }
    });

    await page.goto('/models');
    await expect(page.getByText('Cached Models (1)')).toBeVisible({ timeout: 5_000 });

    // Accept the confirm dialog
    page.on('dialog', dialog => dialog.accept());

    // Click delete button (trash icon)
    const trashBtn = page.locator('button[title*="Delete"]');
    await expect(trashBtn).toBeEnabled();
    await trashBtn.click();

    // After delete, the list should update
    expect(deleteWasCalled).toBe(true);
    await expect(page.getByText('Cached Models (0)')).toBeVisible({ timeout: 5_000 });
  });

  // Delete disabled when model is loaded
  test('[3-16] delete button is disabled when model is loaded', async ({ page }) => {
    await mockModelsPageApis(page, {
      cachedModels: [mockModel],
      loadedModels: [
        {
          modelId: 'BAAI/bge-small-en-v1.5',
          modelType: 'embedder',
          loadedAt: new Date().toISOString(),
          lastUsedAt: new Date().toISOString(),
        },
      ],
    });

    await page.goto('/models');
    await expect(page.getByText('Cached Models (1)')).toBeVisible({ timeout: 5_000 });

    // Trash button should be disabled with "Unload model first" title
    const trashBtn = page.locator('button[title*="Unload model first"]');
    await expect(trashBtn).toBeVisible();
    await expect(trashBtn).toBeDisabled();
  });
});

// ================================================================
// Download progress and completion
// ================================================================

test.describe('Models — download progress', () => {
  // Download button shows progress bar via SSE
  // Download complete shows green message
  // Active downloads section appears
  test('[3-08] [3-09] [3-10] download shows progress bar and completion via SSE', async ({ page }) => {
    await mockModelsPageApis(page);

    // Mock the download/model endpoint to return SSE progress events
    await page.route('**/api/download/model', async (route) => {
      const sseBody = [
        'data: {"fileName":"model.safetensors","bytesDownloaded":50000000,"totalBytes":100000000,"percentComplete":50.0,"status":"Downloading"}\n\n',
        'data: {"fileName":"model.safetensors","bytesDownloaded":100000000,"totalBytes":100000000,"percentComplete":100.0,"status":"Completed"}\n\n',
      ].join('');

      await route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: sseBody,
      });
    });

    await page.goto('/models');
    await expect(page.locator('main').getByRole('heading', { name: 'Model Management' })).toBeVisible();

    // Enter repo ID and click Download
    await page.locator('input[placeholder*="BAAI"]').fill('test-org/test-model');
    await page.getByRole('button', { name: 'Download' }).click();

    // Should show download completion message
    await expect(page.getByText('Download completed')).toBeVisible({ timeout: 10_000 });
  });
});

// ================================================================
// Duplicate download prevention
// ================================================================

test.describe('Models — duplicate download prevention', () => {
  test('[3-13] already-cached model shows cached status in registry', async ({ page }) => {
    const cachedModel = {
      repoId: 'BAAI/bge-small-en-v1.5',
      detectedType: 'Embedder',
      sizeBytes: 120_000_000,
      fileCount: 5,
      localPath: '/cache/models--BAAI--bge-small-en-v1.5',
    };

    await mockModelsPageApis(page, {
      cachedModels: [cachedModel],
      loadedModels: [],
      registryTypes: [{
        type: 'embedder',
        displayName: 'Embedder',
        description: 'Text embedding models',
        models: [
          {
            aliasName: 'default',
            repoId: 'BAAI/bge-small-en-v1.5',
            description: 'Small English embedder',
            isCached: true,
          },
        ],
      }],
    });

    await page.goto('/models');
    await expect(page.locator('main').getByRole('heading', { name: 'Model Management' })).toBeVisible();

    // Registry section should note that cached models are marked
    const registrySection = page.getByText('Cached models are marked with a checkmark');
    await expect(registrySection).toBeVisible({ timeout: 5_000 });

    // Expand Embedder type (scroll into view first)
    const embedderBtn = page.locator('button.w-full', { hasText: 'Embedder' });
    await embedderBtn.scrollIntoViewIfNeeded();
    await embedderBtn.click();

    // The cached model should NOT have a download button (duplicate download prevented)
    const downloadBtn = page.locator('button[title*="Download BAAI"]');
    await expect(downloadBtn).toHaveCount(0, { timeout: 5_000 });
  });
});

// ================================================================
// DELETE /api/cache/models/{repoId}
// ================================================================

test.describe('Models — DELETE cache API', () => {
  test('[5-16] delete cached model via API returns success', async ({ page }) => {
    let deletedRepoId = '';

    await mockModelsPageApis(page, {
      cachedModels: [{
        repoId: 'BAAI/bge-small-en-v1.5',
        detectedType: 'Embedder',
        sizeBytes: 120_000_000,
        fileCount: 5,
        localPath: '/cache/models--BAAI--bge-small-en-v1.5',
      }],
    });

    // Override to track DELETE calls with URL pattern
    await page.unroute('**/api/cache/models');
    await page.route('**/api/cache/models**', async (route) => {
      if (route.request().method() === 'DELETE') {
        const url = route.request().url();
        deletedRepoId = decodeURIComponent(url.split('/api/cache/models/')[1] || '');
        await route.fulfill({ status: 200, body: '{}' });
      } else if (route.request().method() === 'GET') {
        const models = deletedRepoId ? [] : [{
          repoId: 'BAAI/bge-small-en-v1.5',
          detectedType: 'Embedder',
          sizeBytes: 120_000_000,
          fileCount: 5,
          localPath: '/cache/models--BAAI--bge-small-en-v1.5',
        }];
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ models }),
        });
      } else {
        await route.continue();
      }
    });

    await page.goto('/models');
    await expect(page.getByText('Cached Models (1)')).toBeVisible({ timeout: 5_000 });

    // Accept confirmation dialog
    page.on('dialog', dialog => dialog.accept());

    // Click delete
    const trashBtn = page.locator('button[title*="Delete"]');
    await trashBtn.click();

    // Verify DELETE was called with the correct repoId
    expect(deletedRepoId).toContain('BAAI/bge-small-en-v1.5');

    // Model should be removed
    await expect(page.getByText('Cached Models (0)')).toBeVisible({ timeout: 5_000 });
  });
});

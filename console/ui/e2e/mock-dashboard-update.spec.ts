import { test, expect } from './fixtures/base.fixture';
import { mockDashboardApis } from './fixtures/api-mocks';

// ================================================================
// Dashboard — empty cached models + Registry download + Update check
// Covers: empty cached, registry download, update check
// ================================================================

// ================================================================
// Dashboard shows empty cached state
// ================================================================

test.describe('Dashboard — empty cached models', () => {
  test('[2-09] shows empty message when no models cached', async ({ page }) => {
    await mockDashboardApis(page, { cachedModels: [], loadedModels: [] });

    await page.goto('/');
    await expect(page.locator('main').getByRole('heading', { name: 'System Status' }).first()).toBeVisible({ timeout: 5_000 });

    // Cached Models section should show 0 or empty state
    // Dashboard shows "Cached Models" heading with the list
    await expect(page.getByText('Cached Models')).toBeVisible();

    // With no models, should show "No models cached" and "(0)"
    await expect(page.getByText('No models cached', { exact: true })).toBeVisible({ timeout: 5_000 });
  });
});

// ================================================================
// Download from registry
// ================================================================

test.describe('Models — registry download', () => {
  test('[3-12] clicking download button on uncached registry model triggers download', async ({ page }) => {
    // Mock all models page APIs
    await page.route('**/api/cache/models', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ models: [] }),
      });
    });
    await page.route('**/api/cache/loaded', async (route) => {
      await route.fulfill({ status: 200, contentType: 'application/json', body: '[]' });
    });
    await page.route('**/api/registry/models', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          modelTypes: [{
            type: 'embedder',
            displayName: 'Embedder',
            description: 'Text embedding models',
            models: [
              {
                aliasName: 'default',
                repoId: 'BAAI/bge-small-en-v1.5',
                description: 'Small English embedder',
                isCached: false,
              },
            ],
          }],
        }),
      });
    });

    // Mock download endpoint to succeed quickly
    let downloadTriggered = false;
    await page.route('**/api/download/model', async (route) => {
      downloadTriggered = true;
      await route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: 'data: {"fileName":"model.safetensors","bytesDownloaded":100,"totalBytes":100,"percentComplete":100.0,"status":"Completed"}\n\n',
      });
    });

    await page.goto('/models');
    await expect(page.locator('main').getByRole('heading', { name: 'Model Management' })).toBeVisible();

    // Expand Embedder type (click the expand button, not the option or label)
    await page.locator('button.w-full', { hasText: 'Embedder' }).click();

    // Should show the model with "Not downloaded" status
    await expect(page.getByText('Not downloaded')).toBeVisible({ timeout: 5_000 });

    // Click the download button (download icon in the registry row)
    const downloadBtn = page.locator('button[title*="Download BAAI"]');
    await expect(downloadBtn).toBeVisible();
    await downloadBtn.click();

    // Download should have been triggered (mock resolves synchronously)
    await page.waitForResponse(resp => resp.url().includes('/api/download/model'));
    expect(downloadTriggered).toBe(true);
  });
});

// ================================================================
// Update check shows download button
// ================================================================

test.describe('Layout — update check', () => {
  test('shows update button when newer version available', async ({ page }) => {
    // Mock version endpoint
    await page.route('**/api/system/version', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ version: '0.13.4' }),
      });
    });

    // Mock update check to return available update
    await page.route('**/api/system/update', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          currentVersion: '0.13.4',
          latestVersion: '0.14.0',
          updateAvailable: true,
          releaseUrl: 'https://github.com/example/release',
          downloadUrl: 'https://github.com/example/download',
          assetSize: 50_000_000,
        }),
      });
    });

    await page.goto('/');
    await expect(page.locator('main')).toBeVisible();

    // The sidebar footer should show "v0.14.0" button
    await expect(page.getByRole('button', { name: /v0\.14\.0/ })).toBeVisible({ timeout: 10_000 });
  });
});

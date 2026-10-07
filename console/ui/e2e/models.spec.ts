import { test, expect } from './fixtures/base.fixture';

test.describe('Models page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/models');
    await expect(page.getByRole('heading', { name: /Model Management/ })).toBeVisible();
  });

  // ================================================================
  // 14.1 Cached Models
  // ================================================================

  // Cached models table with columns
  test('cached models section shows table with columns', async ({ page }) => {
    const heading = page.getByRole('heading', { name: /Cached Models/ });
    await expect(heading).toBeVisible();

    // If models exist, verify table structure
    const headingText = await heading.textContent();
    const count = parseInt(headingText?.match(/\((\d+)\)/)?.[1] ?? '0');

    if (count > 0) {
      // Table headers should be present
      await expect(page.getByRole('columnheader', { name: 'Repository' })).toBeVisible();
      await expect(page.getByRole('columnheader', { name: 'Type' })).toBeVisible();
      await expect(page.getByRole('columnheader', { name: 'Size' })).toBeVisible();
      await expect(page.getByRole('columnheader', { name: 'Files' })).toBeVisible();
      await expect(page.getByRole('columnheader', { name: 'Status' })).toBeVisible();
    }
  });

  // Status badges (Ready/Loaded/Downloading)
  test('cached models show status badges', async ({ page }) => {
    // Check if there are cached models
    const heading = page.getByRole('heading', { name: /Cached Models/ });
    const headingText = await heading.textContent();
    const count = parseInt(headingText?.match(/\((\d+)\)/)?.[1] ?? '0');

    if (count > 0) {
      // At least one status badge should be visible (Ready or Loaded)
      const statusBadges = page.locator('td').filter({ hasText: /Ready|Loaded|Downloading/ });
      await expect(statusBadges.first()).toBeVisible();
    }
  });

  // Empty state
  test('empty cached models shows message or table when populated', async ({ page }) => {
    // Wait for the refresh icon to stop spinning (API calls completed)
    const refreshBtn = page.getByRole('button', { name: /Refresh/ });
    await expect(refreshBtn.locator('.animate-spin')).toBeHidden({ timeout: 15_000 });

    // Now read the count after data has loaded
    const heading = page.getByRole('heading', { name: /Cached Models/ });
    await expect(heading).toContainText(/\(\d+\)/, { timeout: 5_000 });
    const headingText = await heading.textContent();
    const match = headingText?.match(/\((\d+)\)/);
    const count = match ? parseInt(match[1]) : -1;

    if (count === 0) {
      await expect(page.getByText(/No models in cache|no cached models/i)).toBeVisible();
    } else if (count > 0) {
      // Table should be visible when models exist
      await expect(page.locator('table').first()).toBeVisible();
    }
    // If count === -1 (no match), test is inconclusive — skip silently
  });

  // ================================================================
  // 14.4 Model Registry
  // ================================================================

  // Registry shows all domain types
  test('model registry shows type list with counts', async ({ page }) => {
    const registryHeading = page.getByRole('heading', { name: /Model Registry/ });
    await expect(registryHeading).toBeVisible();

    const registryText = await registryHeading.textContent();
    expect(registryText).toMatch(/Model Registry \(\d+ types\)/);

    // Should have expandable type buttons
    const typeButtons = page.locator('button').filter({ hasText: /models$/ });
    const count = await typeButtons.count();
    expect(count).toBeGreaterThan(0);
  });

  // Clicking type expands alias table; Alias info displayed
  test('clicking a type expands its model table', async ({ page }) => {
    // Find and click the first type button (e.g., Embedder)
    const typeButton = page.locator('button').filter({ hasText: /models$/ }).first();
    await typeButton.click();

    // Table should expand — check for registry-specific columns
    await expect(page.getByRole('columnheader', { name: 'Alias' })).toBeVisible();
    await expect(page.getByRole('columnheader', { name: 'Repository ID' })).toBeVisible();
    await expect(page.getByRole('columnheader', { name: 'Description' })).toBeVisible();
    await expect(page.getByRole('columnheader', { name: 'Actions' }).last()).toBeVisible();
  });

  // Alias names shown (not empty/undefined)
  test('registry alias column shows actual alias names', async ({ page }) => {
    // Expand first type
    const typeButton = page.locator('button').filter({ hasText: /models$/ }).first();
    await typeButton.click();

    // Wait for table to appear
    await expect(page.getByRole('columnheader', { name: 'Alias' })).toBeVisible();

    // Get all alias cells (they have bg-primary/10 styling)
    const aliasCells = page.locator('span.rounded.text-xs.font-medium');
    const count = await aliasCells.count();
    expect(count).toBeGreaterThan(0);

    // Each alias should NOT be empty or "undefined"
    for (let i = 0; i < count; i++) {
      const text = await aliasCells.nth(i).textContent();
      expect(text).toBeTruthy();
      expect(text).not.toBe('undefined');
      expect(text).not.toBe('');
    }
  });

  // Cache status display (Cached/Not downloaded)
  test('registry shows cached/not-downloaded status', async ({ page }) => {
    // Expand first type
    const typeButton = page.locator('button').filter({ hasText: /models$/ }).first();
    await typeButton.click();

    // Should see at least one status indicator
    const statusIndicators = page.getByText(/Cached|Not downloaded/);
    await expect(statusIndicators.first()).toBeVisible();
  });

  // Collapse type
  test('clicking expanded type collapses it', async ({ page }) => {
    // Expand first type
    const typeButton = page.locator('button').filter({ hasText: /models$/ }).first();
    await typeButton.click();

    // Verify expanded (Alias column visible)
    await expect(page.getByRole('columnheader', { name: 'Alias' })).toBeVisible();

    // Click again to collapse
    await typeButton.click();

    // Alias column should be hidden
    await expect(page.getByRole('columnheader', { name: 'Alias' })).toBeHidden();
  });

  // ================================================================
  // 14.5 Refresh
  // ================================================================

  // Refresh button
  test('refresh button reloads all sections', async ({ page }) => {
    const refreshButton = page.getByRole('button', { name: /Refresh/ });
    await expect(refreshButton).toBeVisible();

    // Track API calls
    const apiCalls: string[] = [];
    page.on('request', (req) => {
      const url = req.url();
      if (url.includes('/api/cache/models') || url.includes('/api/cache/loaded') || url.includes('/api/registry')) {
        apiCalls.push(url);
      }
    });

    await refreshButton.click();

    // Wait for all 3 API calls to complete
    await expect.poll(() => apiCalls.length, { timeout: 5_000 }).toBeGreaterThanOrEqual(3);

    // Should have made API calls for cache, loaded, and registry
    expect(apiCalls.some(u => u.includes('/api/cache/models'))).toBe(true);
    expect(apiCalls.some(u => u.includes('/api/cache/loaded'))).toBe(true);
    expect(apiCalls.some(u => u.includes('/api/registry'))).toBe(true);
  });
});

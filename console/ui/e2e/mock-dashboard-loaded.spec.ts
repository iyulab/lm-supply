import { test, expect } from './fixtures/base.fixture';
import { mockDashboardApis } from './fixtures/api-mocks';

// ================================================================
// Dashboard — loaded models display + refresh polling
// Covers: loaded models detail, loaded models refresh
// ================================================================

// ================================================================
// Dashboard shows loaded models with details
// ================================================================

test.describe('Dashboard — loaded models', () => {
  test('[2-10] shows loaded model details (model ID, type, last used)', async ({ page }) => {
    await mockDashboardApis(page, {
      cachedModels: [],
      loadedModels: [
        {
          modelId: 'BAAI/bge-small-en-v1.5',
          modelType: 'embedder',
          loadedAt: '2026-02-26T10:00:00.000Z',
          lastUsedAt: '2026-02-26T10:30:00.000Z',
        },
      ],
    });

    await page.goto('/');
    await expect(page.locator('main').getByRole('heading', { name: 'System Status' }).first()).toBeVisible({ timeout: 5_000 });

    // Loaded Models section should show count
    await expect(page.getByText('Loaded Models (1)')).toBeVisible({ timeout: 5_000 });

    // Should display model ID in the loaded models section
    await expect(page.getByText('BAAI/bge-small-en-v1.5').first()).toBeVisible();

    // Should display model type badge
    await expect(page.locator('main').getByText('embedder')).toBeVisible();
  });
});

// ================================================================
// Dashboard loaded models refresh via polling
// ================================================================

test.describe('Dashboard — loaded models refresh', () => {
  test('dashboard polls loaded models and reflects updated data', async ({ page }) => {
    // Track how many times the loaded endpoint is called
    let loadedCallCount = 0;
    let returnTwo = false;

    await page.route('**/api/cache/models', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ models: [] }),
      });
    });

    // Start with 1 loaded model, then switch to 2 via flag
    await page.route('**/api/cache/loaded', async (route) => {
      loadedCallCount++;
      const models = returnTwo
        ? [
            {
              modelId: 'BAAI/bge-small-en-v1.5',
              modelType: 'embedder',
              loadedAt: '2026-02-26T10:00:00.000Z',
              lastUsedAt: '2026-02-26T10:30:00.000Z',
            },
            {
              modelId: 'sentence-transformers/all-MiniLM-L6-v2',
              modelType: 'embedder',
              loadedAt: '2026-02-26T10:05:00.000Z',
              lastUsedAt: '2026-02-26T10:35:00.000Z',
            },
          ]
        : [
            {
              modelId: 'BAAI/bge-small-en-v1.5',
              modelType: 'embedder',
              loadedAt: '2026-02-26T10:00:00.000Z',
              lastUsedAt: '2026-02-26T10:30:00.000Z',
            },
          ];

      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(models),
      });
    });

    await page.goto('/');
    await expect(page.locator('main').getByRole('heading', { name: 'System Status' }).first()).toBeVisible({ timeout: 5_000 });

    // Initially should show 1 loaded model
    await expect(page.getByText('Loaded Models (1)')).toBeVisible({ timeout: 5_000 });

    // Switch mock to return 2 models — next poll will pick this up
    returnTwo = true;

    // Wait for polling to update (polls every 5s)
    await expect(page.getByText('Loaded Models (2)')).toBeVisible({ timeout: 10_000 });

    // Verify polling called the endpoint multiple times
    expect(loadedCallCount).toBeGreaterThanOrEqual(2);
  });
});

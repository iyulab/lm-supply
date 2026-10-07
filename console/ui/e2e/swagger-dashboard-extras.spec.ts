import { test, expect } from './fixtures/base.fixture';
import { mockDashboardApis } from './fixtures/api-mocks';

// ================================================================
// Swagger + Dashboard CPU-only
// Covers: Swagger UI, CPU-only dashboard
// ================================================================

test.describe('Swagger UI', () => {
  // Swagger page accessible
  test('/swagger page loads Swagger UI', async ({ page }) => {
    await page.goto('/swagger');

    // API docs page should render with API heading
    await expect(page.getByRole('heading', { name: /LMSupply Console API/i })).toBeVisible({ timeout: 10_000 });
  });
});

test.describe('Dashboard — CPU-only state', () => {
  // GPU unavailable shows appropriate state
  test('dashboard shows CPU-only state when GPU unavailable', async ({ page }) => {
    // Mock with CPU-only system status
    await mockDashboardApis(page, {
      status: {
        engineReady: true,
        gpuAvailable: false,
        gpuProvider: 'Cpu',
        gpuName: '',
        cpuUsage: 0.15,
        ramTotalMB: 16384,
        ramUsageMB: 4096,
        ramUsagePercent: 25,
        processMemoryMB: 200,
        timestamp: new Date().toISOString(),
      },
      cachedModels: [],
      loadedModels: [],
    });

    await page.goto('/');
    await expect(page.getByRole('heading', { name: /Dashboard/ })).toBeVisible();

    // GPU should show as unavailable or CPU
    await expect(page.getByText(/CPU|Not Available|No GPU/i)).toBeVisible({ timeout: 5_000 });

    // VRAM card should not show values or show N/A
    const vramText = page.getByText(/VRAM/i);
    if (await vramText.count() > 0) {
      // If VRAM section exists, it should show N/A or 0
      await expect(page.getByText(/N\/A|0 MB|Not available/i)).toBeVisible();
    }
  });
});

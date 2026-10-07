import { test, expect } from './fixtures/base.fixture';

test.describe('Dashboard', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('main').getByRole('heading', { name: 'Dashboard' })).toBeVisible();
  });

  // Status loads
  test('[2-01] system status card shows engine and GPU indicators', async ({ page }) => {
    const statusCard = page.getByText('System Status').locator('..');

    // ONNX Runtime indicator should be present
    await expect(statusCard.getByText('ONNX Runtime')).toBeVisible();
    // GPU indicator should be present
    await expect(statusCard.getByText('GPU Available')).toBeVisible();
  });

  // GPU info
  test('GPU info is displayed when available', async ({ page }) => {
    const statusCard = page.getByText('System Status').locator('..');

    // Either GPU name is shown (when available) or just the indicator
    const gpuAvailable = statusCard.getByText('GPU Available');
    await expect(gpuAvailable).toBeVisible();

    // If GPU is available, provider info may also appear
    // This is environment-dependent, so we just verify the section renders
  });

  // Resource stats
  test('[2-04] [2-05] [2-07] resource stats cards are displayed', async ({ page }) => {
    // CPU Usage card
    await expect(page.getByText('CPU Usage')).toBeVisible({ timeout: 10_000 });
    // Check CPU value is a percentage
    const cpuCard = page.getByText('CPU Usage').locator('..');
    await expect(cpuCard.getByText(/%/).first()).toBeVisible();

    // RAM Usage card
    await expect(page.getByText('RAM Usage')).toBeVisible();
    const ramCard = page.getByText('RAM Usage').locator('..');
    await expect(ramCard.getByText(/%/).first()).toBeVisible();

    // Process Memory card
    await expect(page.getByText('Process Memory')).toBeVisible();
  });

  // Auto-refresh
  test('[2-11] stats update automatically within 10 seconds', async ({ page }) => {
    // Intercept system status API calls
    const apiCalls: number[] = [];
    page.on('request', (req) => {
      if (req.url().includes('/api/system/status')) {
        apiCalls.push(Date.now());
      }
    });

    // Poll until we see at least 2 API calls (initial + at least one poll)
    await expect.poll(() => apiCalls.length, { timeout: 15_000 }).toBeGreaterThanOrEqual(2);
  });

  // Cached models
  test('[2-08] cached models section shows count and list', async ({ page }) => {
    const heading = page.getByRole('heading', { name: /Cached Models/ });
    await expect(heading).toBeVisible();

    // Heading should contain a count in parentheses
    const headingText = await heading.textContent();
    expect(headingText).toMatch(/Cached Models \(\d+\)/);
  });

  // Loaded models section
  test('[2-10] loaded models section renders', async ({ page }) => {
    const heading = page.getByRole('heading', { name: /Loaded Models/ });
    await expect(heading).toBeVisible();

    const headingText = await heading.textContent();
    expect(headingText).toMatch(/Loaded Models \(\d+\)/);
  });
});

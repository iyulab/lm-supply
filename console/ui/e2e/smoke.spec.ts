import { test, expect } from './fixtures/base.fixture';

test.describe('Smoke tests', () => {
  test('health endpoint returns healthy', async ({ request }) => {
    const response = await request.get('/health');
    expect(response.ok()).toBe(true);
    const body = await response.json();
    expect(body.status).toBe('healthy');
  });

  test('home page loads and renders sidebar', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('nav')).toBeVisible();
    // Dashboard heading should render
    await expect(page.getByText('System Status')).toBeVisible({ timeout: 10_000 });
  });

  test('API registry endpoint returns model types', async ({ request }) => {
    const response = await request.get('/api/registry/models');
    expect(response.ok()).toBe(true);
    const body = await response.json();
    expect(body.modelTypes).toBeDefined();
    expect(body.modelTypes.length).toBeGreaterThan(0);
  });
});

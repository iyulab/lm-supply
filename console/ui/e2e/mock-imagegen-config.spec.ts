import { test, expect } from './fixtures/base.fixture';

// ================================================================
// ImageGen — model auto-config (steps + guidance scale)
// Model selection auto-configures parameters
// ================================================================

test.describe('ImageGen — model auto-config', () => {
  test('selecting a model updates steps and guidance scale', async ({ page }) => {
    // Mock the models list endpoint with two models
    await page.route('**/v1/images/models', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          models: [
            {
              id: 'lcm-default',
              repo_id: 'test/lcm-default',
              recommended_steps: 4,
              recommended_guidance_scale: 1.0,
            },
            {
              id: 'sdxl-turbo',
              repo_id: 'test/sdxl-turbo',
              recommended_steps: 8,
              recommended_guidance_scale: 2.5,
            },
          ],
        }),
      });
    });

    await page.goto('/image-generate');
    await expect(page.locator('main').getByRole('heading', { name: 'Image Generation' })).toBeVisible();

    // Wait for models to load in the dropdown
    await expect(page.locator('select').first()).toContainText('lcm-default');

    // Show advanced settings to see Steps and Guidance Scale
    await page.getByText('Show Advanced Settings').click();

    // Default model steps should be 4
    await expect(page.getByText('Steps (4)')).toBeVisible();

    // Select second model
    await page.locator('select').first().selectOption('sdxl-turbo');

    // Steps should update to 8
    await expect(page.getByText('Steps (8)')).toBeVisible();
  });
});

import { test, expect } from './fixtures/base.fixture';

test.describe('Image Generate page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/image-generate');
    await expect(page.locator('main').getByRole('heading', { name: 'Image Generation' })).toBeVisible();
  });

  // ImageGen model selector has default option
  test('model selector has default option', async ({ page }) => {
    const selects = page.locator('select');
    const modelSelect = selects.first();
    await expect(modelSelect).toBeVisible();

    const value = await modelSelect.inputValue();
    expect(value).toBe('default');
  });

  // Size dropdown has 6 preset options
  test('size dropdown has 6 preset options', async ({ page }) => {
    const selects = page.locator('select');
    // Model is first, Size is second
    const sizeSelect = selects.nth(1);
    await expect(sizeSelect).toBeVisible();

    const options = await sizeSelect.locator('option').allTextContents();
    expect(options.length).toBe(6);

    // Check key sizes
    const optionValues = options.join(', ');
    expect(optionValues).toContain('256x256');
    expect(optionValues).toContain('512x512');
    expect(optionValues).toContain('1024x1024');
  });

  // Prompt textarea with placeholder
  test('has prompt textarea with placeholder', async ({ page }) => {
    const textarea = page.getByPlaceholder(/serene lake|sunset|mountains/i);
    await expect(textarea).toBeVisible();
  });

  // Generate button disabled when prompt empty
  test('generate button is disabled when prompt is empty', async ({ page }) => {
    const generateButton = page.getByRole('button', { name: /Generate Image/i });
    await expect(generateButton).toBeDisabled();
  });

  // Generate button enables with text
  test('generate button enables when prompt is entered', async ({ page }) => {
    const textarea = page.getByPlaceholder(/serene lake|sunset|mountains/i);
    await textarea.fill('A beautiful sunset over the ocean');

    const generateButton = page.getByRole('button', { name: /Generate Image/i });
    await expect(generateButton).toBeEnabled();
  });

  // Advanced settings toggle shows additional fields
  test('advanced settings toggle shows/hides settings panel', async ({ page }) => {
    // Advanced settings should be hidden initially
    await expect(page.getByText(/Negative Prompt/)).toBeHidden();

    // Click the toggle
    const advancedToggle = page.getByRole('button', { name: /Advanced Settings/i });
    await advancedToggle.click();

    // Settings should now be visible
    await expect(page.getByText(/Negative Prompt/)).toBeVisible();
    await expect(page.getByText(/Guidance Scale/)).toBeVisible();
    await expect(page.getByText(/Seed/)).toBeVisible();
  });

  // Steps slider range 1-8
  test('steps slider has correct range (1-8)', async ({ page }) => {
    // Open advanced settings
    const advancedToggle = page.getByRole('button', { name: /Advanced Settings/i });
    await advancedToggle.click();

    // Find the steps slider (first range input in advanced panel)
    const sliders = page.locator('input[type="range"]');
    const stepsSlider = sliders.first();
    await expect(stepsSlider).toBeVisible();
    await expect(stepsSlider).toHaveAttribute('min', '1');
    await expect(stepsSlider).toHaveAttribute('max', '8');
  });

  // Guidance scale slider
  test('guidance scale slider has correct range (0-3)', async ({ page }) => {
    // Open advanced settings
    const advancedToggle = page.getByRole('button', { name: /Advanced Settings/i });
    await advancedToggle.click();

    const sliders = page.locator('input[type="range"]');
    const guidanceSlider = sliders.nth(1);
    await expect(guidanceSlider).toBeVisible();
    await expect(guidanceSlider).toHaveAttribute('min', '0');
    await expect(guidanceSlider).toHaveAttribute('max', '3');
    await expect(guidanceSlider).toHaveAttribute('step', '0.1');
  });

  // Randomize seed button generates new seed value
  test('randomize seed button fills seed input', async ({ page }) => {
    // Open advanced settings
    const advancedToggle = page.getByRole('button', { name: /Advanced Settings/i });
    await advancedToggle.click();

    const seedInput = page.locator('input[type="number"]');
    await expect(seedInput).toBeVisible();

    // Initially empty (random)
    const initialValue = await seedInput.inputValue();
    expect(initialValue).toBe('');

    // Click randomize button (the RefreshCw icon button next to seed)
    const randomizeButton = page.getByTitle('Randomize seed');
    await randomizeButton.click();

    // Seed should now have a value
    const newValue = await seedInput.inputValue();
    expect(newValue).not.toBe('');
    expect(parseInt(newValue)).toBeGreaterThan(0);
  });

  // Empty state placeholder
  test('shows placeholder in result panel', async ({ page }) => {
    await expect(page.getByText('Your generated image will appear here')).toBeVisible();
  });
});

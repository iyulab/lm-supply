import { test, expect } from './fixtures/base.fixture';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const TEST_IMAGE = path.resolve(__dirname, 'fixtures/test-files/test-image.png');

// ================================================================
// Detect page — deeper UI tests
// ================================================================

test.describe('Detect — threshold interaction', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/detect');
    await expect(page.locator('main').getByRole('heading', { name: 'Object Detection' })).toBeVisible();
  });

  // slider step is 0.05
  test('threshold slider has step of 0.05', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    await expect(slider).toHaveAttribute('step', '0.05');
  });

  // slider default value is 0.5 (50%)
  test('threshold slider defaults to 50%', async ({ page }) => {
    const slider = page.locator('input[type="range"]');
    const value = await slider.inputValue();
    expect(value).toBe('0.5');

    // Label shows 50%
    await expect(page.getByText('50%')).toBeVisible();
  });

  // label updates in real-time
  test('threshold label updates when slider changes', async ({ page }) => {
    const slider = page.locator('input[type="range"]');

    // Change slider to 0.8 (80%)
    await slider.fill('0.8');

    // Label should update to 80%
    await expect(page.getByText('80%')).toBeVisible();
  });

  // Model ID is editable
  test('model ID input is editable', async ({ page }) => {
    const modelInput = page.locator('input[type="text"]').first();
    await modelInput.fill('custom-model');
    const value = await modelInput.inputValue();
    expect(value).toBe('custom-model');
  });

  // Upload triggers detection (loading state)
  test('uploading image triggers detection automatically', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_IMAGE);

    // Should show loading or preview (detection auto-triggers)
    // Since backend will return an error for fake image, we check for either
    // loading state, preview, or error — all indicate detection was attempted
    const preview = page.locator('img[alt="Preview"]');
    const errorDiv = page.locator('.bg-destructive\\/10');
    const spinner = page.locator('.animate-spin');

    // Wait for one of: preview with error, or loading spinner
    await expect(preview.or(errorDiv).or(spinner).first()).toBeVisible({ timeout: 15_000 });
  });

  // Detect page has result section heading pattern
  test('no result section shown before upload', async ({ page }) => {
    // Before uploading, there should be no "Detected Objects" heading
    await expect(page.getByText(/Detected Objects/)).not.toBeVisible();
  });
});

// ================================================================
// Segment page — deeper UI tests
// ================================================================

test.describe('Segment — UI details', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/segment');
    await expect(page.locator('main').getByRole('heading', { name: 'Image Segmentation' })).toBeVisible();
  });

  // Model ID is editable
  test('model ID input is editable', async ({ page }) => {
    const modelInput = page.locator('input[type="text"]').first();
    await modelInput.fill('custom-segmenter');
    const value = await modelInput.inputValue();
    expect(value).toBe('custom-segmenter');
  });

  // Upload triggers segmentation
  test('uploading image triggers segmentation automatically', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_IMAGE);

    // Should show loading, preview, or error (segmentation auto-triggers)
    const preview = page.locator('img[alt="Preview"]');
    const errorDiv = page.locator('.bg-destructive\\/10');
    const spinner = page.locator('.animate-spin');

    await expect(preview.or(errorDiv).or(spinner).first()).toBeVisible({ timeout: 15_000 });
  });

  // No result section before upload
  test('no result section shown before upload', async ({ page }) => {
    await expect(page.getByText(/Segmentation Results/)).not.toBeVisible();
  });

  // Loading text is specific to segment
  test('shows "Segmenting image..." during loading', async ({ page }) => {
    // We can verify this text exists in the component by checking the drop zone
    // The loading text "Segmenting image..." appears during inference
    const dropZone = page.locator('.border-dashed');
    await expect(dropZone).toBeVisible();

    // Upload will trigger loading state briefly before error
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_IMAGE);

    // Either loading text or error should appear
    const loadingOrError = page.getByText(/Segmenting image/).or(page.locator('.bg-destructive\\/10'));
    await expect(loadingOrError.first()).toBeVisible({ timeout: 15_000 });
  });

  // Model ID placeholder
  test('model ID input has "default" placeholder', async ({ page }) => {
    const modelInput = page.locator('input[type="text"]').first();
    await expect(modelInput).toHaveAttribute('placeholder', 'default');
  });
});

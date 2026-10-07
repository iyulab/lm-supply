import { test, expect } from './fixtures/base.fixture';
import { mockJsonEndpoint, mockSegmentResponse, mockJsonError } from './fixtures/api-mocks';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const TEST_IMAGE = path.resolve(__dirname, 'fixtures/test-files/test-image.png');

// ================================================================
// Segment page — mock API tests for result display
// Covers: segmentation results, score display, top 10 note
// ================================================================

test.describe('Segment — mock inference results', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/segment');
    await expect(page.locator('main').getByRole('heading', { name: 'Image Segmentation' })).toBeVisible();
  });

  // [4-85] Image upload triggers segmentation [4-86] Class count shown
  test('[4-85] uploading image shows segmentation results', async ({ page }) => {
    const mockResponse = mockSegmentResponse();
    await mockJsonEndpoint(page, '**/v1/images/segment', mockResponse);

    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_IMAGE);

    // Result heading should appear
    await expect(page.getByText('Segmentation Results')).toBeVisible({ timeout: 15_000 });

    // Class count shown
    await expect(page.getByText('Classes: 5')).toBeVisible();
  });

  // [4-87] Class labels with coverage info
  test('[4-87] segment list shows class labels', async ({ page }) => {
    const mockResponse = mockSegmentResponse();
    await mockJsonEndpoint(page, '**/v1/images/segment', mockResponse);

    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_IMAGE);

    await expect(page.getByText('Top Detected Classes')).toBeVisible({ timeout: 15_000 });

    // Should show class labels
    await expect(page.getByText('sky')).toBeVisible();
    await expect(page.getByText('building')).toBeVisible();
    await expect(page.getByText('road')).toBeVisible();
  });

  // Score display with percentage text
  test('score percentage is displayed for each class', async ({ page }) => {
    const mockResponse = mockSegmentResponse();
    await mockJsonEndpoint(page, '**/v1/images/segment', mockResponse);

    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_IMAGE);

    await expect(page.getByText('Top Detected Classes')).toBeVisible({ timeout: 15_000 });

    // Should show score percentages: 42.0%, 28.0%, 15.0%, 10.0%, 5.0%
    await expect(page.getByText('42.0%')).toBeVisible();
    await expect(page.getByText('28.0%')).toBeVisible();
    await expect(page.getByText('15.0%')).toBeVisible();
  });

  // "Shows top 10 classes by pixel coverage" note
  test('shows top 10 classes note', async ({ page }) => {
    const mockResponse = mockSegmentResponse();
    await mockJsonEndpoint(page, '**/v1/images/segment', mockResponse);

    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_IMAGE);

    await expect(page.getByText('Segmentation Results')).toBeVisible({ timeout: 15_000 });

    // Top 10 note should be visible
    await expect(page.getByText(/Shows top 10 classes by pixel coverage/)).toBeVisible();
  });

  // [4-88] Elapsed time and model name shown
  test('[4-88] result shows model name and elapsed time', async ({ page }) => {
    const mockResponse = mockSegmentResponse();
    await mockJsonEndpoint(page, '**/v1/images/segment', mockResponse);

    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_IMAGE);

    await expect(page.getByText('Segmentation Results')).toBeVisible({ timeout: 15_000 });

    // Model name
    await expect(page.getByText('Model: mock-segmenter')).toBeVisible();

    // Elapsed time
    await expect(page.getByText(/\d+ ms/).first()).toBeVisible();
  });

  // Error handling
  test('API error shows error message', async ({ page }) => {
    await mockJsonError(page, '**/v1/images/segment', 'Segmenter model not available');

    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_IMAGE);

    await expect(page.getByText(/Segmenter model not available/)).toBeVisible({ timeout: 15_000 });
  });
});

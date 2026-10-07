import { test, expect } from './fixtures/base.fixture';
import { mockJsonEndpoint, mockDetectResponse, mockJsonError } from './fixtures/api-mocks';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const TEST_IMAGE = path.resolve(__dirname, 'fixtures/test-files/test-image.png');

// ================================================================
// Detect page — mock API tests for result display
// Covers: detection results, class summary, threshold, empty
// ================================================================

test.describe('Detect — mock inference results', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/detect');
    await expect(page.locator('main').getByRole('heading', { name: 'Object Detection' })).toBeVisible();
  });

  // [4-77] Image upload triggers detection [4-78] Detection results summary
  test('[4-77] uploading image shows detected objects list', async ({ page }) => {
    const mockResponse = mockDetectResponse();
    await mockJsonEndpoint(page, '**/v1/images/detect', mockResponse);

    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_IMAGE);

    // Result heading: "Detected Objects (3)"
    await expect(page.getByText(/Detected Objects \(3\)/)).toBeVisible({ timeout: 15_000 });
  });

  // [4-78] Class summary pills with label and count
  test('[4-78] class summary shows labels with counts', async ({ page }) => {
    const mockResponse = mockDetectResponse();
    await mockJsonEndpoint(page, '**/v1/images/detect', mockResponse);

    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_IMAGE);

    await expect(page.getByText(/Detected Objects/)).toBeVisible({ timeout: 15_000 });

    // Should show summary pills: "person: 2", "car: 1"
    await expect(page.getByText('person: 2')).toBeVisible();
    await expect(page.getByText('car: 1')).toBeVisible();
  });

  // [4-79] Detection items show confidence and bounding box
  test('[4-79] detection results show confidence and bounding box', async ({ page }) => {
    const mockResponse = mockDetectResponse();
    await mockJsonEndpoint(page, '**/v1/images/detect', mockResponse);

    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_IMAGE);

    await expect(page.getByText(/Detected Objects/)).toBeVisible({ timeout: 15_000 });

    // Should show confidence percentage (95.0%)
    await expect(page.getByText('95.0%')).toBeVisible();

    // Should show bounding box coordinates
    await expect(page.getByText(/Box:/).first()).toBeVisible();
  });

  // [4-82] Elapsed time shown after detection
  test('[4-82] elapsed time shown after detection', async ({ page }) => {
    const mockResponse = mockDetectResponse();
    await mockJsonEndpoint(page, '**/v1/images/detect', mockResponse);

    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_IMAGE);

    await expect(page.getByText(/Detected Objects/)).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText(/\d+ ms/)).toBeVisible();
  });

  // [4-81] No objects detected shows empty message
  test('[4-81] no objects detected shows empty message', async ({ page }) => {
    const emptyResponse = { id: 'detect-mock', model: 'mock', objects: [] };
    await mockJsonEndpoint(page, '**/v1/images/detect', emptyResponse);

    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_IMAGE);

    await expect(page.getByText(/No objects detected above the confidence threshold/)).toBeVisible({ timeout: 15_000 });
  });

  // Error handling
  test('API error shows error message', async ({ page }) => {
    await mockJsonError(page, '**/v1/images/detect', 'Detector model failed to initialize');

    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_IMAGE);

    await expect(page.getByText(/Detector model failed/)).toBeVisible({ timeout: 15_000 });
  });
});

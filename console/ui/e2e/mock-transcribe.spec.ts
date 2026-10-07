import { test, expect } from './fixtures/base.fixture';
import { mockJsonEndpoint, mockTranscribeResponse, mockJsonError } from './fixtures/api-mocks';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const TEST_AUDIO = path.resolve(__dirname, 'fixtures/test-files/test-audio.wav');

// ================================================================
// Transcribe page — mock API tests for result display
// Covers: file upload → transcription, result display, segments
// ================================================================

test.describe('Transcribe — mock inference results', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/transcribe');
    await expect(page.locator('main').getByRole('heading', { name: 'Speech to Text' })).toBeVisible();
    await expect(page.getByText('Loading models...')).toBeHidden({ timeout: 15_000 });
  });

  // Audio upload triggers transcription result
  test('uploading audio shows transcription result', async ({ page }) => {
    const mockResponse = mockTranscribeResponse();
    await mockJsonEndpoint(page, '**/v1/audio/transcriptions', mockResponse);

    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_AUDIO);

    // Transcription heading should appear (use role to avoid matching text content)
    const resultHeading = page.getByRole('heading', { name: 'Transcription' });
    await expect(resultHeading).toBeVisible({ timeout: 15_000 });

    // Full text should appear in the result card
    const resultCard = page.locator('.bg-card');
    await expect(resultCard.locator('p.whitespace-pre-wrap')).toContainText('Hello, this is a test transcription');
  });

  // Result shows language, duration, elapsed time
  test('result shows language and duration metadata', async ({ page }) => {
    const mockResponse = mockTranscribeResponse();
    await mockJsonEndpoint(page, '**/v1/audio/transcriptions', mockResponse);

    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_AUDIO);

    // Wait for result heading
    await expect(page.getByRole('heading', { name: 'Transcription' })).toBeVisible({ timeout: 15_000 });

    // Language should be shown
    await expect(page.getByText('Language: en')).toBeVisible();

    // Duration should be shown
    await expect(page.getByText(/Duration: 3\.45s/)).toBeVisible();

    // Elapsed time in ms
    await expect(page.getByText(/\d+ ms/).first()).toBeVisible();
  });

  // Segments table shows timestamps and text
  test('segments table shows timestamps and text', async ({ page }) => {
    const mockResponse = mockTranscribeResponse();
    await mockJsonEndpoint(page, '**/v1/audio/transcriptions', mockResponse);

    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_AUDIO);

    // Segments heading should appear
    await expect(page.getByText('Segments')).toBeVisible({ timeout: 15_000 });

    // Should show segment timestamps
    await expect(page.getByText(/\[0\.00s - 1\.50s\]/)).toBeVisible();
    await expect(page.getByText(/\[1\.50s - 3\.45s\]/)).toBeVisible();

    // Segment text in span elements (exact match to avoid matching full text)
    await expect(page.getByText('Hello, this is a test', { exact: true })).toBeVisible();
    await expect(page.getByText('transcription of the audio file.', { exact: true })).toBeVisible();
  });

  // Error handling
  test('API error shows error message', async ({ page }) => {
    await mockJsonError(page, '**/v1/audio/transcriptions', 'Transcriber model not loaded');

    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_AUDIO);

    await expect(page.getByText(/Transcriber model not loaded/)).toBeVisible({ timeout: 15_000 });
  });
});

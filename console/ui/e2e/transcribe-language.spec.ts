import { test, expect } from './fixtures/base.fixture';
import { mockJsonEndpoint } from './fixtures/api-mocks';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const TEST_AUDIO = path.resolve(__dirname, 'fixtures/test-files/test-audio.wav');

// ================================================================
// Transcribe — Korean language + format hints
// Covers: Korean audio, format hints
// ================================================================

test.describe('Transcribe — Korean audio', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/transcribe');
    await expect(page.locator('main').getByRole('heading', { name: 'Speech to Text' })).toBeVisible();
    await expect(page.getByText('Loading models...')).toBeHidden({ timeout: 15_000 });
  });

  // [4-38] Korean audio transcription shows Korean text and language "ko"
  test('[4-38] Korean transcription shows Korean text and ko language', async ({ page }) => {
    const koreanResponse = {
      text: '안녕하세요, 오늘 날씨가 좋습니다.',
      task: 'transcribe',
      language: 'ko',
      duration: 4.2,
      segments: [
        { id: 0, start: 0.0, end: 2.1, text: '안녕하세요,' },
        { id: 1, start: 2.1, end: 4.2, text: '오늘 날씨가 좋습니다.' },
      ],
    };

    await mockJsonEndpoint(page, '**/v1/audio/transcriptions', koreanResponse);

    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_AUDIO);

    // Wait for result
    await expect(page.getByRole('heading', { name: 'Transcription' })).toBeVisible({ timeout: 15_000 });

    // Korean text should display without encoding issues
    await expect(page.getByText('안녕하세요, 오늘 날씨가 좋습니다.')).toBeVisible();

    // Language should show 'ko'
    await expect(page.getByText('Language: ko')).toBeVisible();

    // Segments should show Korean text
    await expect(page.getByText('안녕하세요,', { exact: true })).toBeVisible();
    await expect(page.getByText('오늘 날씨가 좋습니다.', { exact: true })).toBeVisible();
  });
});

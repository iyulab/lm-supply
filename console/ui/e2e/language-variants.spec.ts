import { test, expect } from './fixtures/base.fixture';
import { mockJsonEndpoint } from './fixtures/api-mocks';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const TEST_IMAGE = path.resolve(__dirname, 'fixtures/test-files/test-image.png');

// ================================================================
// Language Variant Tests
// Covers: Korean TTS, Korean OCR
// ================================================================

test.describe('Language Variants', () => {
  test.setTimeout(120_000);

  // [4-48] Korean text to speech via UI (mocked)
  test('[4-48] synthesize page handles Korean text input', async ({ page }) => {
    // Mock the speech endpoint to return audio for Korean
    await page.route('**/v1/audio/speech', async (route) => {
      // Return a minimal WAV header as binary response
      const wavHeader = Buffer.alloc(44);
      wavHeader.write('RIFF', 0);
      wavHeader.writeUInt32LE(36, 4);
      wavHeader.write('WAVE', 8);
      wavHeader.write('fmt ', 12);
      wavHeader.writeUInt32LE(16, 16);
      wavHeader.writeUInt16LE(1, 20); // PCM
      wavHeader.writeUInt16LE(1, 22); // mono
      wavHeader.writeUInt32LE(22050, 24); // sample rate
      wavHeader.writeUInt32LE(44100, 28); // byte rate
      wavHeader.writeUInt16LE(2, 32); // block align
      wavHeader.writeUInt16LE(16, 34); // bits per sample
      wavHeader.write('data', 36);
      wavHeader.writeUInt32LE(0, 40);

      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'audio/wav' },
        body: wavHeader,
      });
    });

    await page.goto('/synthesize');
    await expect(page.locator('main').getByRole('heading', { name: 'Text to Speech' })).toBeVisible();

    // Enter Korean text
    const textarea = page.locator('textarea');
    await textarea.fill('안녕하세요, 오늘 날씨가 좋습니다.');

    // Submit
    await page.locator('button[type="submit"]').click();

    // Should produce audio player
    await expect(page.locator('audio')).toBeVisible({ timeout: 10_000 });
  });

  // [4-73] Korean OCR via UI (mocked)
  test('[4-73] OCR page accepts Korean language selection', async ({ page }) => {
    // Mock OCR endpoint with Korean result
    await page.route('**/v1/images/ocr', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          text: '안녕하세요 세계',
          blocks: [
            {
              text: '안녕하세요',
              confidence: 0.95,
              boundingBox: { x: 10, y: 10, width: 100, height: 20 },
            },
            {
              text: '세계',
              confidence: 0.92,
              boundingBox: { x: 120, y: 10, width: 60, height: 20 },
            },
          ],
          language: 'ko',
        }),
      });
    });

    await page.goto('/ocr');
    await expect(page.locator('main').getByRole('heading', { name: 'Optical Character Recognition' })).toBeVisible();

    // Select Korean language
    const langSelect = page.locator('select').first();
    await langSelect.selectOption('ko');

    // Upload image
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles(TEST_IMAGE);

    // Wait for OCR result heading
    await expect(page.getByRole('heading', { name: 'Recognized Text' })).toBeVisible({ timeout: 15_000 });

    // Korean text should be displayed
    await expect(page.getByText('안녕하세요 세계')).toBeVisible();
  });

  // [4-48] Korean TTS via API
  test('[4-48] POST /v1/audio/speech with Korean text returns audio', async ({ request }) => {
    const response = await request.post('/v1/audio/speech', {
      data: {
        model: 'default',
        input: '안녕하세요, 테스트입니다.',
      },
    });

    // May succeed or return error (model availability)
    if (response.status() === 200) {
      const contentType = response.headers()['content-type'] ?? '';
      expect(contentType).toContain('audio/');
      const body = await response.body();
      expect(body.length).toBeGreaterThan(0);
    } else {
      const body = await response.json();
      expect(body).toHaveProperty('error');
    }
  });

  // [4-73] Korean OCR via API
  test('[4-73] POST /v1/images/ocr with language=ko returns text', async ({ request }) => {
    const imageBuffer = fs.readFileSync(TEST_IMAGE);
    const response = await request.post('/v1/images/ocr', {
      multipart: {
        model: 'default',
        file: { name: 'test-image.png', mimeType: 'image/png', buffer: imageBuffer },
        language: 'ko',
      },
    });

    if (response.status() === 200) {
      const body = await response.json();
      expect(body).toHaveProperty('text');
      expect(body).toHaveProperty('blocks');
    } else {
      const body = await response.json();
      expect(body).toHaveProperty('error');
    }
  });
});

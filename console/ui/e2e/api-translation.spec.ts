import { test, expect } from '@playwright/test';
import { BASE, requireCachedModel } from './host-capabilities';
import { KOREAN_COUNT, KOREAN_HELLO, KOREAN_THANKS } from './fixtures';

// The registered translation models translate into English (the default one from Korean).
const KOREAN = KOREAN_HELLO;

// ================================================================
// POST /v1/translate (Custom API)
// ================================================================
test.describe('Custom Translation API', () => {
  test('basic translation returns translated text', async ({ request }) => {
    await requireCachedModel(request, 'translator');
    const resp = await request.post(`${BASE}/v1/translate`, {
      data: { input: KOREAN, model: 'default' },
    });
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(body.translations.length).toBe(1);
    expect(body.translations[0].source_text).toBe(KOREAN);
    expect(body.translations[0].translated_text.length).toBeGreaterThan(0);
  });

  test('missing text field returns 400', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/translate`, {
      data: { source_language: 'en', target_language: 'ko', model: 'default' },
    });
    expect(resp.status()).toBe(400);
  });

  test('invalid language code returns appropriate error', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/translate`, {
      data: { text: 'Hello', source_language: 'xx', target_language: 'zz', model: 'default' },
    });
    // Should return 400 or 404 (not 200 or 422)
    expect(resp.status()).not.toBe(200);
    expect(resp.status()).not.toBe(422);
  });
});

// ================================================================
// POST /v2/translate (DeepL-compatible)
// ================================================================
test.describe('DeepL-compatible Translation', () => {
  test('response format matches DeepL: {translations:[{detected_source_language,text}]}', async ({ request }) => {
    await requireCachedModel(request, 'translator');
    const resp = await request.post(`${BASE}/v2/translate`, {
      data: { text: [KOREAN, KOREAN_THANKS], source_lang: 'KO', target_lang: 'EN' },
    });
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(Array.isArray(body.translations)).toBe(true);
    expect(body.translations.length).toBe(2);
    expect(body.translations[0]).toHaveProperty('detected_source_language');
    expect(body.translations[0]).toHaveProperty('text');
  });

  test('uppercase target_lang code is handled (KO not ko)', async ({ request }) => {
    const resp = await request.post(`${BASE}/v2/translate`, {
      data: { text: ['Hello'], target_lang: 'KO' },
    });
    expect(resp.status()).not.toBe(422);
    expect(resp.status()).not.toBe(400);
  });

  test('single string input (not array) accepted', async ({ request }) => {
    const resp = await request.post(`${BASE}/v2/translate`, {
      data: { text: 'Hello', target_lang: 'KO' },
    });
    expect(resp.status()).not.toBe(422);
  });

  test('array input returns same-length translations array', async ({ request }) => {
    await requireCachedModel(request, 'translator');
    const resp = await request.post(`${BASE}/v2/translate`, {
      data: { text: KOREAN_COUNT, source_lang: 'KO', target_lang: 'EN' },
    });
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(body.translations.length).toBe(3);
  });

  test('missing target_lang returns 400', async ({ request }) => {
    const resp = await request.post(`${BASE}/v2/translate`, {
      data: { text: ['Hello'] },
    });
    expect(resp.status()).toBe(400);
  });

  test('detected_source_language is uppercase', async ({ request }) => {
    await requireCachedModel(request, 'translator');
    const resp = await request.post(`${BASE}/v2/translate`, {
      data: { text: [KOREAN], source_lang: 'KO', target_lang: 'EN' },
    });
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    const lang = body.translations[0].detected_source_language;
    expect(lang).toBe(lang.toUpperCase());
  });
});

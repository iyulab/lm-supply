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
      data: { model: 'default' },
    });
    expect(resp.status()).toBe(400);
  });

  // The registered models are language pairs; a name that is none of them is a 404 that lists the ones there are.
  test('a model that is no registered translation pair returns 404 listing the known ones', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/translate`, {
      data: { input: 'Hello', model: 'no-such-pair' },
    });
    expect(resp.status()).toBe(404);
    const body = await resp.json();
    expect(body.error.code).toBe('model_not_found');
    expect(body.error.message).toContain('ko-en');
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

  // The registered models translate into English, so the pair that exists is KO to EN; the codes are read in either case.
  test('language codes are read in lowercase as well as uppercase', async ({ request }) => {
    await requireCachedModel(request, 'translator');
    const upper = await request.post(`${BASE}/v2/translate`, {
      data: { text: [KOREAN], source_lang: 'KO', target_lang: 'EN' },
    });
    const lower = await request.post(`${BASE}/v2/translate`, {
      data: { text: [KOREAN], source_lang: 'ko', target_lang: 'en' },
    });
    expect(upper.status()).toBe(200);
    expect(lower.status()).toBe(200);
    expect((await lower.json()).translations[0].text).toBe((await upper.json()).translations[0].text);
  });

  test('single string input (not array) accepted', async ({ request }) => {
    await requireCachedModel(request, 'translator');
    const resp = await request.post(`${BASE}/v2/translate`, {
      data: { text: KOREAN, source_lang: 'KO', target_lang: 'EN' },
    });
    expect(resp.status()).toBe(200);
    expect((await resp.json()).translations).toHaveLength(1);
  });

  test('a target language no registered model translates into is 404 model_not_found', async ({ request }) => {
    const resp = await request.post(`${BASE}/v2/translate`, {
      data: { text: ['Hello'], target_lang: 'KO' },
    });
    expect(resp.status()).toBe(404);
    expect((await resp.json()).error.code).toBe('model_not_found');
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

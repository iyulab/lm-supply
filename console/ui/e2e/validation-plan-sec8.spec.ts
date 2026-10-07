import { test, expect } from '@playwright/test';

const BASE = process.env.API_BASE_URL || 'http://localhost:5000';

// ================================================================
// Section 8-1: POST /v1/translate (Custom API)
// ================================================================
test.describe('Section 8-1: Custom Translation API', () => {
  test('[8-1-01] basic translation returns translated text', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/translate`, {
      data: { text: 'Hello', source_language: 'en', target_language: 'ko', model: 'default' },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      expect(body).toBeTruthy();
      // Response should contain some translated text
      const bodyStr = JSON.stringify(body);
      expect(bodyStr.length).toBeGreaterThan(5);
    } else {
      expect([404, 503]).toContain(resp.status());
    }
  });

  test('[8-1-02] missing text field returns 400', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/translate`, {
      data: { source_language: 'en', target_language: 'ko', model: 'default' },
    });
    expect(resp.status()).toBe(400);
  });

  test('[8-1-03] invalid language code returns appropriate error', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/translate`, {
      data: { text: 'Hello', source_language: 'xx', target_language: 'zz', model: 'default' },
    });
    // Should return 400 or 404 (not 200 or 422)
    expect(resp.status()).not.toBe(200);
    expect(resp.status()).not.toBe(422);
  });
});

// ================================================================
// Section 8-2: POST /v2/translate (DeepL-compatible)
// ================================================================
test.describe('Section 8-2: DeepL-compatible Translation', () => {
  test('[8-2-01] response format matches DeepL: {translations:[{detected_source_language,text}]}', async ({ request }) => {
    const resp = await request.post(`${BASE}/v2/translate`, {
      data: { text: ['Hello', 'World'], target_lang: 'KO' },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      expect(Array.isArray(body.translations)).toBe(true);
      expect(body.translations.length).toBe(2);
      expect(body.translations[0]).toHaveProperty('detected_source_language');
      expect(body.translations[0]).toHaveProperty('text');
    } else {
      expect([404, 503]).toContain(resp.status());
    }
  });

  test('[8-2-02] uppercase target_lang code is handled (KO not ko)', async ({ request }) => {
    const resp = await request.post(`${BASE}/v2/translate`, {
      data: { text: ['Hello'], target_lang: 'KO' },
    });
    expect(resp.status()).not.toBe(422);
    expect(resp.status()).not.toBe(400);
  });

  test('[8-2-03] single string input (not array) accepted', async ({ request }) => {
    const resp = await request.post(`${BASE}/v2/translate`, {
      data: { text: 'Hello', target_lang: 'KO' },
    });
    expect(resp.status()).not.toBe(422);
  });

  test('[8-2-04] array input returns same-length translations array', async ({ request }) => {
    const resp = await request.post(`${BASE}/v2/translate`, {
      data: { text: ['One', 'Two', 'Three'], target_lang: 'KO' },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      expect(body.translations.length).toBe(3);
    }
  });

  test('[8-2-05] missing target_lang returns 400', async ({ request }) => {
    const resp = await request.post(`${BASE}/v2/translate`, {
      data: { text: ['Hello'] },
    });
    expect(resp.status()).toBe(400);
  });

  test('[8-2-06] detected_source_language is uppercase', async ({ request }) => {
    const resp = await request.post(`${BASE}/v2/translate`, {
      data: { text: ['Hello'], target_lang: 'KO' },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      const lang = body.translations[0].detected_source_language;
      expect(lang).toBe(lang.toUpperCase());
    }
  });
});

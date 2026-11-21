import { test, expect } from '@playwright/test';

const BASE = process.env.API_BASE_URL || 'http://localhost:5000';

// ================================================================
// System Status and Monitoring
// ================================================================
test.describe('System API', () => {
  test('GET /api/system/status returns required fields', async ({ request }) => {
    const resp = await request.get(`${BASE}/api/system/status`);
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    // The metrics sit in `status`, next to the models that are loaded.
    expect(body).toHaveProperty('loadedModels');
    expect(body).toHaveProperty('models');
    expect(body.status).toHaveProperty('engineReady');
    expect(body.status).toHaveProperty('gpuAvailable');
    expect(body.status).toHaveProperty('cpuUsage');
    expect(body.status).toHaveProperty('ramUsageMB');
    expect(body.status).toHaveProperty('ramTotalMB');
    expect(body.status).toHaveProperty('processMemoryMB');
  });

  test('GET /api/system/gpu returns GPU info or null', async ({ request }) => {
    const resp = await request.get(`${BASE}/api/system/gpu`);
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    // Without a GPU the host still names what it computes on ("CPU Only") and says no GPU is available.
    expect(typeof body.isAvailable).toBe('boolean');
    expect(typeof body.name).toBe('string');
    expect(typeof body.provider).toBe('string');
  });

  test('GET /api/system/memory returns memory metrics', async ({ request }) => {
    const resp = await request.get(`${BASE}/api/system/memory`);
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(body).toBeTruthy();
    // Should have some memory-related fields
    const bodyStr = JSON.stringify(body).toLowerCase();
    const hasMemoryField = bodyStr.includes('mb') || bodyStr.includes('memory') ||
      bodyStr.includes('ram') || bodyStr.includes('bytes');
    expect(hasMemoryField).toBe(true);
  });

  test('GET /api/system/version returns version and rid', async ({ request }) => {
    const resp = await request.get(`${BASE}/api/system/version`);
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(body).toHaveProperty('version');
    expect(body).toHaveProperty('rid');
  });

  test('GET /api/system/update endpoint exists (not 404)', async ({ request }) => {
    const resp = await request.get(`${BASE}/api/system/update`);
    expect(resp.status()).not.toBe(404);
    expect(resp.status()).not.toBe(405);
  });

  test('GET /api/system/metrics/stream uses text/event-stream', async () => {
    // The stream never ends: read the headers and the first event, then close the connection
    const controller = new AbortController();
    try {
      const resp = await fetch(`${BASE}/api/system/metrics/stream`, { signal: controller.signal });
      expect(resp.status).toBe(200);
      expect(resp.headers.get('content-type')).toContain('text/event-stream');
      const first = await resp.body!.getReader().read();
      expect(new TextDecoder().decode(first.value)).toMatch(/^data: /);
    } finally {
      controller.abort();
    }
  });
});

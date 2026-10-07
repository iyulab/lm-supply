import { test, expect } from '@playwright/test';

const BASE = process.env.API_BASE_URL || 'http://localhost:5000';

// ================================================================
// Section 14-1: System Status and Monitoring
// ================================================================
test.describe('Section 14-1: System API', () => {
  test('[14-1-01] GET /api/system/status returns required fields', async ({ request }) => {
    const resp = await request.get(`${BASE}/api/system/status`);
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(body).toHaveProperty('engineReady');
    expect(body).toHaveProperty('gpuAvailable');
    expect(body).toHaveProperty('cpuUsage');
    expect(body).toHaveProperty('ramUsageMB');
    expect(body).toHaveProperty('ramTotalMB');
    expect(body).toHaveProperty('processMemoryMB');
  });

  test('[14-1-02] GET /api/system/gpu returns GPU info or null', async ({ request }) => {
    const resp = await request.get(`${BASE}/api/system/gpu`);
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    // Either GPU info or null fields when no GPU
    if (body.gpuName !== null) {
      expect(body).toHaveProperty('gpuName');
      expect(body).toHaveProperty('gpuProvider');
    }
  });

  test('[14-1-03] GET /api/system/memory returns memory metrics', async ({ request }) => {
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

  test('[14-1-04] GET /api/system/version returns version and rid', async ({ request }) => {
    const resp = await request.get(`${BASE}/api/system/version`);
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(body).toHaveProperty('version');
    expect(body).toHaveProperty('rid');
  });

  test('[14-1-05] GET /api/system/update endpoint exists (not 404)', async ({ request }) => {
    const resp = await request.get(`${BASE}/api/system/update`);
    expect(resp.status()).not.toBe(404);
    expect(resp.status()).not.toBe(405);
  });

  test('[14-1-06] GET /api/system/metrics/stream uses text/event-stream', async ({ request }) => {
    // Make the request and check content-type without consuming full stream
    const resp = await request.get(`${BASE}/api/system/metrics/stream`);
    if (resp.status() === 200) {
      const ct = resp.headers()['content-type'];
      expect(ct).toContain('text/event-stream');
    } else {
      // Endpoint might not be implemented — should exist
      expect(resp.status()).not.toBe(404);
      expect(resp.status()).not.toBe(405);
    }
  });
});

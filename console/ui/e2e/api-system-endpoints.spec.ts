import { test, expect } from './fixtures/base.fixture';

// ================================================================
// API System Endpoints — direct API tests via Playwright request context
// ================================================================

test.describe('API System Endpoints', () => {
  // [5-01] GET /health returns healthy status
  test('[5-01] GET /health returns healthy status', async ({ request }) => {
    const response = await request.get('/health');
    expect(response.status()).toBe(200);

    const body = await response.json();
    expect(body.status).toBe('healthy');
  });

  // [5-02] GET /api/system/status returns system info
  test('[5-02] GET /api/system/status returns system metrics', async ({ request }) => {
    const response = await request.get('/api/system/status');
    expect(response.status()).toBe(200);

    const body = await response.json();
    // Status is nested under body.status
    const status = body.status;
    expect(status).toHaveProperty('engineReady');
    expect(status).toHaveProperty('gpuAvailable');
    expect(typeof status.engineReady).toBe('boolean');
    expect(typeof status.gpuAvailable).toBe('boolean');

    // CPU and RAM
    expect(status.cpuUsage).toBeGreaterThanOrEqual(0);
    expect(status.ramTotalMB).toBeGreaterThan(0);
    expect(status.ramUsageMB).toBeGreaterThan(0);
  });

  // [5-03] GET /api/system/gpu returns GPU info
  test('[5-03] GET /api/system/gpu returns GPU information', async ({ request }) => {
    const response = await request.get('/api/system/gpu');
    expect(response.status()).toBe(200);

    const body = await response.json();
    expect(body).toHaveProperty('provider');
    expect(typeof body.provider).toBe('string');
  });

  // [5-04] GET /api/system/version returns version
  test('[5-04] GET /api/system/version returns version and rid', async ({ request }) => {
    const response = await request.get('/api/system/version');
    expect(response.status()).toBe(200);

    const body = await response.json();
    expect(body).toHaveProperty('version');
    expect(typeof body.version).toBe('string');
    expect(body.version).toMatch(/^\d+\.\d+\.\d+/);
  });

  // [5-05] GET /api/system/update returns update info
  test('[5-05] GET /api/system/update returns update availability', async ({ request }) => {
    const response = await request.get('/api/system/update');
    expect(response.status()).toBe(200);

    const body = await response.json();
    expect(body).toHaveProperty('updateAvailable');
    expect(typeof body.updateAvailable).toBe('boolean');
    expect(body).toHaveProperty('latestVersion');
    expect(body).toHaveProperty('currentVersion');
  });

  // [5-06] GET /api/system/metrics/stream returns SSE stream
  test('[5-06] GET /api/system/metrics/stream returns SSE events', async ({ page }) => {
    // Navigate first so page has a valid origin for relative URLs
    await page.goto('/');

    // Use page.evaluate with fetch + AbortController to read first SSE event
    const result = await page.evaluate(async () => {
      const controller = new AbortController();
      const response = await fetch('/api/system/metrics/stream', {
        headers: { 'Accept': 'text/event-stream' },
        signal: controller.signal,
      });

      const contentType = response.headers.get('content-type') ?? '';
      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let text = '';

      // Read until we get at least one complete SSE event
      for (let i = 0; i < 10; i++) {
        const { value, done } = await reader.read();
        if (done) break;
        text += decoder.decode(value, { stream: true });
        if (text.includes('data:')) break;
      }

      controller.abort();
      return { contentType, text };
    });

    expect(result.contentType).toContain('text/event-stream');
    expect(result.text).toContain('data:');
  });
});

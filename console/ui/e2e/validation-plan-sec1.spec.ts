import { test, expect } from '@playwright/test';

const BASE = process.env.API_BASE_URL || 'http://localhost:5000';

// ================================================================
// Section 1-1: Server Startup
// ================================================================
test.describe('Section 1-1: Server Startup', () => {
  test('[1-1-02] Swagger UI accessible', async ({ request }) => {
    const resp = await request.get(`${BASE}/swagger/index.html`);
    expect(resp.status()).toBe(200);
  });

  test('[1-1-03] React UI accessible at root', async ({ request }) => {
    const resp = await request.get(`${BASE}/`);
    expect(resp.status()).toBe(200);
    const body = await resp.text();
    expect(body).toContain('html');
  });

  test('[1-1-04] CORS header present for cross-origin request', async ({ request }) => {
    const resp = await request.get(`${BASE}/v1/models`, {
      headers: { 'Origin': 'http://localhost:5173' },
    });
    expect(resp.status()).toBe(200);
    const headers = resp.headers();
    expect(headers['access-control-allow-origin']).toBeTruthy();
  });
});

// ================================================================
// Section 1-2: ErrorMiddleware
// ================================================================
test.describe('Section 1-2: ErrorMiddleware', () => {
  test('[1-2-01] error response has envelope format {error:{message,type,code}}', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/embeddings`, {
      data: { model: 'nonexistent-model-validation-plan', input: 'test' },
    });
    expect([400, 404, 422, 503]).toContain(resp.status());
    const body = await resp.json();
    expect(body).toHaveProperty('error');
    expect(body.error).toHaveProperty('message');
    expect(body.error).toHaveProperty('type');
  });

  test('[1-2-02] model not found → 404, type: model_not_found or invalid_request_error', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/embeddings`, {
      data: { model: 'nonexistent-model-validation-plan', input: 'test' },
    });
    expect(resp.status()).toBe(404);
    const body = await resp.json();
    expect(body.error).toHaveProperty('code');
    // model_not_found or invalid_request_error with model_not_found code
    expect(['model_not_found', 'invalid_request_error']).toContain(body.error.type);
  });

  test('[1-2-03] missing required field → 400, type: invalid_request_error', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/embeddings`, {
      data: { model: 'default' }, // missing input
    });
    expect(resp.status()).toBe(400);
    const body = await resp.json();
    expect(body.error.type).toBe('invalid_request_error');
  });

  test('[1-2-04] internal error → 500 without stack trace', async ({ request }) => {
    // Empty JSON body to trigger internal error path
    const resp = await request.post(`${BASE}/v1/embeddings`, {
      headers: { 'Content-Type': 'application/json' },
      data: '{}',
    });
    if (resp.status() === 500) {
      const body = await resp.json();
      expect(body.error.message).toBe('An internal server error occurred.');
      // Stack trace must not be exposed
      expect(JSON.stringify(body)).not.toContain('at ');
      expect(JSON.stringify(body)).not.toContain('System.');
    }
    // 400 is also acceptable for empty body
    expect([400, 500]).toContain(resp.status());
  });

  test('[1-2-05] all responses include X-Request-Id header', async ({ request }) => {
    const resp = await request.get(`${BASE}/v1/models`);
    expect(resp.headers()['x-request-id']).toBeTruthy();
  });

  test('[1-2-06] X-Request-Id echoes client-provided value', async ({ request }) => {
    const myId = 'validation-plan-my-id-123';
    const resp = await request.get(`${BASE}/v1/models`, {
      headers: { 'X-Request-Id': myId },
    });
    expect(resp.headers()['x-request-id']).toBe(myId);
  });

  test('[1-2-07] server auto-generates UUID X-Request-Id when not provided', async ({ request }) => {
    const resp = await request.get(`${BASE}/v1/models`);
    const reqId = resp.headers()['x-request-id'];
    expect(reqId).toBeTruthy();
    // UUID v4 pattern: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
    expect(reqId).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i);
  });
});

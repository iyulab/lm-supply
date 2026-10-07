import { test, expect } from '@playwright/test';

const BASE = process.env.API_BASE_URL || 'http://localhost:5000';

// ================================================================
// Section 3-1: Basic Chat Completions
// ================================================================
test.describe('Section 3-1: Basic Chat Completions', () => {
  test('[3-1-01] response has required OpenAI fields', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: {
        model: 'default',
        messages: [{ role: 'user', content: 'Say hello.' }],
        max_completion_tokens: 10,
      },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      expect(body).toHaveProperty('id');
      expect(body.object).toBe('chat.completion');
      expect(typeof body.created).toBe('number');
      expect(body).toHaveProperty('model');
      expect(body.choices[0].message).toHaveProperty('content');
      expect(body).toHaveProperty('usage');
    } else {
      expect([404, 503]).toContain(resp.status());
    }
  });

  test('[3-1-02] finish_reason is "stop"', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: {
        model: 'default',
        messages: [{ role: 'user', content: 'Say one word.' }],
        max_completion_tokens: 5,
      },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      expect(body.choices[0].finish_reason).toBe('stop');
    } else {
      expect([404, 503]).toContain(resp.status());
    }
  });

  test('[3-1-03] empty messages returns 400', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: { model: 'default', messages: [] },
    });
    expect(resp.status()).toBe(400);
    const body = await resp.json();
    expect(body.error.type).toBe('invalid_request_error');
  });

  test('[3-1-04] system + user messages accepted', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: {
        model: 'default',
        messages: [
          { role: 'system', content: 'You are a helpful assistant.' },
          { role: 'user', content: 'Hello.' },
        ],
        max_completion_tokens: 5,
      },
    });
    expect([200, 404, 503]).toContain(resp.status());
    expect(resp.status()).not.toBe(400);
    expect(resp.status()).not.toBe(422);
  });

  test('[3-1-05] max_completion_tokens parameter accepted', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: {
        model: 'default',
        messages: [{ role: 'user', content: 'Hi.' }],
        max_completion_tokens: 5,
      },
    });
    expect(resp.status()).not.toBe(422);
  });

  test('[3-1-06] max_tokens (legacy) parameter accepted', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: {
        model: 'default',
        messages: [{ role: 'user', content: 'Hi.' }],
        max_tokens: 5,
      },
    });
    expect(resp.status()).not.toBe(422);
  });

  test('[3-1-07] temperature and top_p parameters accepted', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: {
        model: 'default',
        messages: [{ role: 'user', content: 'Hi.' }],
        temperature: 0.7,
        top_p: 0.9,
        max_tokens: 5,
      },
    });
    expect(resp.status()).not.toBe(422);
  });

  test('[3-1-08] stop sequences parameter accepted', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: {
        model: 'default',
        messages: [{ role: 'user', content: 'Count to 10.' }],
        stop: ['\n'],
        max_tokens: 20,
      },
    });
    expect(resp.status()).not.toBe(422);
  });
});

// ================================================================
// Section 3-2: Streaming
// ================================================================
test.describe('Section 3-2: Chat Streaming', () => {
  test('[3-2-01] streaming response uses text/event-stream', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: {
        model: 'default',
        messages: [{ role: 'user', content: 'Count to 3.' }],
        stream: true,
        max_tokens: 20,
      },
    });
    if (resp.status() === 200) {
      const ct = resp.headers()['content-type'];
      expect(ct).toContain('text/event-stream');
    } else {
      expect([404, 503]).toContain(resp.status());
    }
  });

  test('[3-2-03] streaming response ends with [DONE]', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: {
        model: 'default',
        messages: [{ role: 'user', content: 'Say hi.' }],
        stream: true,
        max_tokens: 5,
      },
    });
    if (resp.status() === 200) {
      const text = await resp.text();
      expect(text).toContain('data: [DONE]');
    } else {
      expect([404, 503]).toContain(resp.status());
    }
  });
});

// ================================================================
// Section 3-3: Multiple Choices (n > 1)
// ================================================================
test.describe('Section 3-3: Multiple Choices', () => {
  test('[3-3-01] n=3 returns 3 choices', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: {
        model: 'default',
        messages: [{ role: 'user', content: 'Say one word.' }],
        n: 3,
        max_tokens: 5,
      },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      expect(body.choices.length).toBe(3);
    } else {
      expect([404, 503]).toContain(resp.status());
    }
  });

  test('[3-3-02] n=3 choices have indices 0, 1, 2', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: {
        model: 'default',
        messages: [{ role: 'user', content: 'Say one word.' }],
        n: 3,
        max_tokens: 5,
      },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      const indices = body.choices.map((c: { index: number }) => c.index);
      expect(indices).toContain(0);
      expect(indices).toContain(1);
      expect(indices).toContain(2);
    }
  });
});

// ================================================================
// Section 3-4: Tool Calls
// ================================================================
test.describe('Section 3-4: Tool Calls', () => {
  const TOOLS = [{
    type: 'function',
    function: {
      name: 'get_weather',
      description: 'Get current weather',
      parameters: {
        type: 'object',
        properties: { location: { type: 'string' } },
        required: ['location'],
      },
    },
  }];

  test('[3-4-01] tools parameter accepted (not 422)', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: {
        model: 'default',
        messages: [{ role: 'user', content: 'What is the weather in Seoul?' }],
        tools: TOOLS,
        tool_choice: 'auto',
        max_tokens: 50,
      },
    });
    expect(resp.status()).not.toBe(422);
  });

  test('[3-4-06] tool_choice: none returns text response without tool_calls', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: {
        model: 'default',
        messages: [{ role: 'user', content: 'What is the weather in Seoul?' }],
        tools: TOOLS,
        tool_choice: 'none',
        max_tokens: 30,
      },
    });
    if (resp.status() === 200) {
      const body = await resp.json();
      // tool_choice: none → no tool_calls
      expect(body.choices[0].message.tool_calls ?? null).toBeNull();
    } else {
      expect([404, 503]).toContain(resp.status());
    }
  });
});

// ================================================================
// Section 3-5: Response Format
// ================================================================
test.describe('Section 3-5: Response Format', () => {
  test('[3-5-01] response_format json_object accepted', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: {
        model: 'default',
        messages: [{ role: 'user', content: 'Return a JSON object with key "result".' }],
        response_format: { type: 'json_object' },
        max_tokens: 30,
      },
    });
    expect(resp.status()).not.toBe(422);
  });

  test('[3-5-03] invalid response_format type returns 400 or is ignored', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: {
        model: 'default',
        messages: [{ role: 'user', content: 'Hello.' }],
        response_format: { type: 'invalid_format_xyz' },
        max_tokens: 5,
      },
    });
    // Either 400 (rejected) or 200/404/503 (ignored)
    expect(resp.status()).not.toBe(422);
  });
});

// ================================================================
// Section 3-6: Multimodal (image_url)
// ================================================================
test.describe('Section 3-6: Multimodal Vision', () => {
  // 1x1 red pixel PNG as base64
  const PNG_B64 = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg==';

  test('[3-6-01] base64 image_url accepted in messages (not 422)', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: {
        model: 'default',
        messages: [{
          role: 'user',
          content: [
            { type: 'text', text: 'Describe this image.' },
            { type: 'image_url', image_url: { url: PNG_B64 } },
          ],
        }],
        max_tokens: 20,
      },
    });
    expect(resp.status()).not.toBe(422);
  });

  test('[3-6-02] localhost URL in image_url accepted (not 422)', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: {
        model: 'default',
        messages: [{
          role: 'user',
          content: [
            { type: 'text', text: 'Describe.' },
            { type: 'image_url', image_url: { url: `${BASE}/favicon.ico` } },
          ],
        }],
        max_tokens: 20,
      },
    });
    expect(resp.status()).not.toBe(422);
  });
});

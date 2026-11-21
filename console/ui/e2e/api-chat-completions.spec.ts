import { test, expect } from '@playwright/test';
import { PNG_DATA_URL } from './fixtures';
import { BASE, requireAnyCachedModel } from './host-capabilities';

// ================================================================
// Basic Chat Completions
// ================================================================
test.describe('Basic Chat Completions', () => {
  test('response has required OpenAI fields', async ({ request }) => {
    await requireAnyCachedModel(request, 'generator');
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: {
        model: 'default',
        messages: [{ role: 'user', content: 'Say hello.' }],
        max_completion_tokens: 10,
      },
    });
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(body).toHaveProperty('id');
    expect(body.object).toBe('chat.completion');
    expect(typeof body.created).toBe('number');
    expect(body).toHaveProperty('model');
    expect(body.choices[0].message).toHaveProperty('content');
    expect(body).toHaveProperty('usage');
  });

  test('finish_reason is "stop"', async ({ request }) => {
    await requireAnyCachedModel(request, 'generator');
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: {
        model: 'default',
        messages: [{ role: 'user', content: 'Say one word.' }],
        max_completion_tokens: 5,
      },
    });
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(body.choices[0].finish_reason).toBe('stop');
  });

  test('empty messages returns 400', async ({ request }) => {
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: { model: 'default', messages: [] },
    });
    expect(resp.status()).toBe(400);
    const body = await resp.json();
    expect(body.error.type).toBe('invalid_request_error');
  });

  test('system + user messages accepted', async ({ request }) => {
    await requireAnyCachedModel(request, 'generator');
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
    expect(resp.status()).toBe(200);
  });

  test('max_completion_tokens parameter accepted', async ({ request }) => {
    await requireAnyCachedModel(request, 'generator');
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: {
        model: 'default',
        messages: [{ role: 'user', content: 'Hi.' }],
        max_completion_tokens: 5,
      },
    });
    expect(resp.status()).toBe(200);
  });

  test('max_tokens (legacy) parameter accepted', async ({ request }) => {
    await requireAnyCachedModel(request, 'generator');
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: {
        model: 'default',
        messages: [{ role: 'user', content: 'Hi.' }],
        max_tokens: 5,
      },
    });
    expect(resp.status()).toBe(200);
  });

  test('temperature and top_p parameters accepted', async ({ request }) => {
    await requireAnyCachedModel(request, 'generator');
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: {
        model: 'default',
        messages: [{ role: 'user', content: 'Hi.' }],
        temperature: 0.7,
        top_p: 0.9,
        max_tokens: 5,
      },
    });
    expect(resp.status()).toBe(200);
  });

  test('stop sequences parameter accepted', async ({ request }) => {
    await requireAnyCachedModel(request, 'generator');
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: {
        model: 'default',
        messages: [{ role: 'user', content: 'Count to 10.' }],
        stop: ['\n'],
        max_tokens: 20,
      },
    });
    expect(resp.status()).toBe(200);
  });
});

// ================================================================
// Streaming
// ================================================================
test.describe('Chat Streaming', () => {
  test('streaming response uses text/event-stream', async ({ request }) => {
    await requireAnyCachedModel(request, 'generator');
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: {
        model: 'default',
        messages: [{ role: 'user', content: 'Count to 3.' }],
        stream: true,
        max_tokens: 20,
      },
    });
    expect(resp.status()).toBe(200);
    const ct = resp.headers()['content-type'];
    expect(ct).toContain('text/event-stream');
  });

  test('streaming response ends with [DONE]', async ({ request }) => {
    await requireAnyCachedModel(request, 'generator');
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: {
        model: 'default',
        messages: [{ role: 'user', content: 'Say hi.' }],
        stream: true,
        max_tokens: 5,
      },
    });
    expect(resp.status()).toBe(200);
    const text = await resp.text();
    expect(text).toContain('data: [DONE]');
  });
});

// ================================================================
// Multiple Choices (n > 1)
// ================================================================
test.describe('Multiple Choices', () => {
  test('n=3 returns 3 choices', async ({ request }) => {
    await requireAnyCachedModel(request, 'generator');
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: {
        model: 'default',
        messages: [{ role: 'user', content: 'Say one word.' }],
        n: 3,
        max_tokens: 5,
      },
    });
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    expect(body.choices.length).toBe(3);
  });

  test('n=3 choices have indices 0, 1, 2', async ({ request }) => {
    await requireAnyCachedModel(request, 'generator');
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: {
        model: 'default',
        messages: [{ role: 'user', content: 'Say one word.' }],
        n: 3,
        max_tokens: 5,
      },
    });
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    const indices = body.choices.map((c: { index: number }) => c.index);
    expect(indices).toContain(0);
    expect(indices).toContain(1);
    expect(indices).toContain(2);
  });
});

// ================================================================
// Tool Calls
// ================================================================
test.describe('Tool Calls', () => {
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

  test('tools parameter accepted', async ({ request }) => {
    await requireAnyCachedModel(request, 'generator');
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: {
        model: 'default',
        messages: [{ role: 'user', content: 'What is the weather in Seoul?' }],
        tools: TOOLS,
        tool_choice: 'auto',
        max_tokens: 50,
      },
    });
    expect(resp.status()).toBe(200);
  });

  test('tool_choice: none returns text response without tool_calls', async ({ request }) => {
    await requireAnyCachedModel(request, 'generator');
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: {
        model: 'default',
        messages: [{ role: 'user', content: 'What is the weather in Seoul?' }],
        tools: TOOLS,
        tool_choice: 'none',
        max_tokens: 30,
      },
    });
    expect(resp.status()).toBe(200);
    const body = await resp.json();
    // tool_choice: none → no tool_calls
    expect(body.choices[0].message.tool_calls ?? null).toBeNull();
  });
});

// ================================================================
// Response Format
// ================================================================
test.describe('Response Format', () => {
  test('response_format json_object accepted', async ({ request }) => {
    await requireAnyCachedModel(request, 'generator');
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: {
        model: 'default',
        messages: [{ role: 'user', content: 'Return a JSON object with key "result".' }],
        response_format: { type: 'json_object' },
        max_tokens: 30,
      },
    });
    expect(resp.status()).toBe(200);
  });

  test('an unknown response_format type is ignored', async ({ request }) => {
    await requireAnyCachedModel(request, 'generator');
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: {
        model: 'default',
        messages: [{ role: 'user', content: 'Hello.' }],
        response_format: { type: 'invalid_format_xyz' },
        max_tokens: 5,
      },
    });
    // The host reads the types it knows (json_object) and leaves the rest alone.
    expect(resp.status()).toBe(200);
  });
});

// ================================================================
// Multimodal (image_url)
// ================================================================
test.describe('Multimodal Vision', () => {
  test('base64 image_url accepted in messages', async ({ request }) => {
    await requireAnyCachedModel(request, 'generator');
    const resp = await request.post(`${BASE}/v1/chat/completions`, {
      data: {
        model: 'default',
        messages: [{
          role: 'user',
          content: [
            { type: 'text', text: 'Describe this image.' },
            { type: 'image_url', image_url: { url: PNG_DATA_URL } },
          ],
        }],
        max_tokens: 20,
      },
    });
    expect(resp.status()).toBe(200);
  });

  test('localhost URL in image_url accepted', async ({ request }) => {
    await requireAnyCachedModel(request, 'generator');
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
    expect(resp.status()).toBe(200);
  });
});

import { test, expect, type APIRequestContext } from '@playwright/test';

export const BASE = process.env.API_BASE_URL || 'http://localhost:5000';

/**
 * Conditions that decide, before a request is sent, whether the host under test can be expected to serve it.
 *
 * A test that needs a model calls one of the `require*` functions first: when the model is not available on the
 * host the test is skipped, and the skip (with its reason) shows in the report. The test body then asserts the
 * response unconditionally. The response of the request under test never decides whether its assertions run.
 */

interface RegistryModel {
  aliasName: string;
  repoId: string;
  isCached: boolean;
}

async function registryModels(request: APIRequestContext, type: string): Promise<RegistryModel[]> {
  const resp = await request.get(`${BASE}/api/registry/models/${type}`);
  expect(resp.status(), `GET /api/registry/models/${type}`).toBe(200);
  const body = await resp.json();
  return body.models as RegistryModel[];
}

/** Skips the test unless the registry lists the alias of the model type as present in the local model cache. */
export async function requireCachedModel(request: APIRequestContext, type: string, alias = 'default'): Promise<void> {
  const model = (await registryModels(request, type)).find(m => m.aliasName === alias);
  test.skip(!model?.isCached, `${type} model '${alias}' is not in the local model cache on this host`);
}

/**
 * Skips the test unless at least one alias of the model type is cached. For types whose 'default' is picked by the
 * hardware at load time (text generation), so that the registry cannot name its repository.
 */
export async function requireAnyCachedModel(request: APIRequestContext, type: string): Promise<void> {
  const models = await registryModels(request, type);
  test.skip(!models.some(m => m.isCached), `no ${type} model is in the local model cache on this host`);
}

/**
 * Skips the test unless the files an OCR request reads are in the local model cache: the default detector and the recognizer of
 * the language (the registry's OCR entries say per model whether its own files are cached, not only whether its repository is).
 */
export async function requireCachedOcr(request: APIRequestContext, recognizer: string): Promise<void> {
  const models = await registryModels(request, 'ocr');
  for (const alias of ['dbnet-v5', recognizer]) {
    const model = models.find(m => m.aliasName === alias);
    test.skip(!model?.isCached, `OCR model '${alias}' is not in the local model cache on this host`);
  }
}

/** Skips the test unless the repository is in the local model cache (for a model the registry listing does not name, such as a `gguf:` alias). */
export async function requireCachedRepo(request: APIRequestContext, repoId: string, what: string): Promise<void> {
  const resp = await request.get(`${BASE}/api/cache/models`);
  expect(resp.status(), 'GET /api/cache/models').toBe(200);
  const body = await resp.json();
  const cached = (body.models as { repoId: string }[]).some(m => m.repoId.toLowerCase() === repoId.toLowerCase());
  test.skip(!cached, `${what} (${repoId}) is not in the local model cache on this host`);
}

/** Skips the test unless the Hugging Face Hub answers from this machine (the host reads model metadata from it). */
export async function requireHub(request: APIRequestContext): Promise<void> {
  let reachable = false;
  try {
    const resp = await request.get('https://huggingface.co/api/models/BAAI/bge-small-en-v1.5', { timeout: 10_000 });
    reachable = resp.ok();
  } catch {
    reachable = false;
  }
  test.skip(!reachable, 'the Hugging Face Hub is not reachable from this machine');
}

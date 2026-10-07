import { test, expect } from './fixtures/base.fixture';

test.describe('Models page — Load/Unload & Download', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/models');
    await expect(page.getByRole('heading', { name: /Model Management/ })).toBeVisible();
  });

  // ================================================================
  // 14.3 Loaded Models & Pre-load
  // ================================================================

  // Empty state
  test('[3-28] loaded models section renders with count', async ({ page }) => {
    const heading = page.getByRole('heading', { name: /Loaded Models/ });
    await expect(heading).toBeVisible();

    const headingText = await heading.textContent();
    expect(headingText).toMatch(/Loaded Models \(\d+\)/);

    const count = parseInt(headingText?.match(/\((\d+)\)/)?.[1] ?? '0');
    if (count === 0) {
      await expect(page.getByText(/No models currently loaded/)).toBeVisible();
    }
  });

  // Pre-load form elements
  test('[3-19] pre-load form has type selector, model ID input, and load button', async ({ page }) => {
    // Type select
    const typeSelect = page.locator('select').filter({ hasText: 'Generator' });
    await expect(typeSelect).toBeVisible();

    // All 11 type options
    const options = await typeSelect.locator('option').allTextContents();
    expect(options).toContain('Generator');
    expect(options).toContain('Embedder');
    expect(options).toContain('Reranker');
    expect(options).toContain('Transcriber');
    expect(options).toContain('Synthesizer');
    expect(options).toContain('Translator');
    expect(options).toContain('Captioner');
    expect(options).toContain('OCR');
    expect(options).toContain('Detector');
    expect(options).toContain('Segmenter');
    expect(options).toContain('Image Generator');

    // Model ID input
    const modelInput = page.getByPlaceholder('default');
    await expect(modelInput).toBeVisible();

    // Load button
    const loadButton = page.getByRole('button', { name: /Load/ }).filter({ hasText: 'Load' });
    await expect(loadButton).toBeVisible();
  });

  // Advanced options toggle
  test('[3-23] advanced options toggle shows/hides provider and thread inputs', async ({ page }) => {
    const settingsButton = page.getByTitle('Advanced options');
    await expect(settingsButton).toBeVisible();

    // Click to show
    await settingsButton.click();

    // Provider select should appear
    await expect(page.getByText('Provider').first()).toBeVisible();
    await expect(page.getByText('Threads').first()).toBeVisible();

    // Provider should have expected options
    const providerSelect = page.locator('select').filter({ hasText: 'Auto' });
    const providerOptions = await providerSelect.locator('option').allTextContents();
    expect(providerOptions).toContain('Auto');
    expect(providerOptions).toContain('CUDA');
    expect(providerOptions).toContain('DirectML');
    expect(providerOptions).toContain('CPU');

    // Click to hide
    await settingsButton.click();
    // Provider label should be hidden (within the collapsed section)
    await expect(page.locator('label').filter({ hasText: 'Provider' })).toBeHidden();
  });

  // Generator-specific advanced options
  test('[3-24] generator type shows context and concurrency options', async ({ page }) => {
    // Open advanced options
    await page.getByTitle('Advanced options').click();

    // With "generator" selected, Max Context and Concurrency should appear
    await expect(page.getByText('Max Context')).toBeVisible();
    await expect(page.getByText('Concurrency')).toBeVisible();
  });

  // Embedder-specific advanced options
  test('[3-25] embedder type shows max sequence length option', async ({ page }) => {
    // Change type to embedder
    const typeSelect = page.locator('select').filter({ hasText: 'Generator' });
    await typeSelect.selectOption('embedder');

    // Open advanced options
    await page.getByTitle('Advanced options').click();

    // Max Seq Length should appear
    await expect(page.getByText('Max Seq Length')).toBeVisible();

    // Generator-specific options should NOT appear
    await expect(page.getByText('Max Context')).toBeHidden();
    await expect(page.getByText('Concurrency')).toBeHidden();
  });

  // Load error for non-existent model
  test('[3-26] loading non-existent model shows error', async ({ page }) => {
    // Enter a non-existent model ID
    const modelInput = page.getByPlaceholder('default');
    await modelInput.fill('nonexistent-model-that-does-not-exist-12345');

    const loadButton = page.getByRole('button', { name: /Load/ }).filter({ hasText: 'Load' });
    await loadButton.click();

    // Wait for error message
    await expect(page.locator('.bg-destructive\\/10, .text-destructive').first()).toBeVisible({ timeout: 30_000 });
  });

  // ================================================================
  // 14.2 Download Section
  // ================================================================

  // Check model (valid repo)
  // Network-dependent: calls backend → HuggingFace API
  test('[3-06] check model shows info for valid repository', async ({ page }) => {
    test.slow(); // triple the default timeout

    const repoInput = page.getByPlaceholder(/e\.g\..*BAAI/);
    await expect(repoInput).toBeVisible();

    const checkButton = page.getByRole('button', { name: 'Check' });
    await expect(checkButton).toBeVisible();

    // Type a known repo
    await repoInput.fill('BAAI/bge-small-en-v1.5');
    await checkButton.click();

    // Should show model found (network-dependent, allow generous timeout)
    const modelFoundText = page.getByText(/Model found/);
    await expect(modelFoundText).toBeVisible({ timeout: 60_000 });
    // Verify model info is displayed near the "Model found" result
    // Use .first() since "Type:" also appears in loaded models section
    await expect(page.getByText(/Type:\s*embedder/).first()).toBeVisible();
  });

  // Check invalid repo
  test('[3-07] check model shows error for invalid repository', async ({ page }) => {
    const repoInput = page.getByPlaceholder(/e\.g\..*BAAI/);
    await repoInput.fill('totally-invalid-repo-does-not-exist/xyz');

    const checkButton = page.getByRole('button', { name: 'Check' });
    await checkButton.click();

    // Should show model not found or error
    await expect(page.getByText(/Model not found|Error/)).toBeVisible({ timeout: 30_000 });
  });

  // Download button exists
  test('[3-08] download button exists in HuggingFace section', async ({ page }) => {
    const downloadButton = page.getByRole('button', { name: 'Download' }).first();
    await expect(downloadButton).toBeVisible();
  });
});

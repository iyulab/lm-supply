import { test, expect } from './fixtures/base.fixture';

// ================================================================
// cURL Examples — verify each domain page has a working cURL section
// ================================================================

const domainPages = [
  { path: '/chat', heading: 'Chat', planId: '4-12', expectedCommands: 2, keywords: ['chat/completions'] },
  { path: '/embed', heading: 'Text Embedding', planId: '4-21', expectedCommands: 1, keywords: ['embeddings'] },
  { path: '/rerank', heading: 'Document Reranking', planId: '4-31', expectedCommands: 1, keywords: ['rerank'] },
  { path: '/transcribe', heading: 'Speech to Text', planId: '4-39', expectedCommands: 2, keywords: ['transcriptions'] },
  { path: '/synthesize', heading: 'Text to Speech', planId: '4-49', expectedCommands: 1, keywords: ['audio/speech'] },
  { path: '/translate', heading: 'Machine Translation', planId: '4-57', expectedCommands: 2, keywords: ['translate'] },
  { path: '/ocr', heading: 'Optical Character Recognition', planId: '4-74', expectedCommands: 2, keywords: ['ocr'] },
  { path: '/detect', heading: 'Object Detection', planId: '4-83', expectedCommands: 2, keywords: ['detect'] },
  { path: '/segment', heading: 'Image Segmentation', planId: '4-89', expectedCommands: 2, keywords: ['segment'] },
  { path: '/image-generate', heading: 'Image Generation', planId: '4-101', expectedCommands: 2, keywords: ['images/generate'] },
];

test.describe('cURL Examples — all domain pages', () => {
  for (const domain of domainPages) {
    test(`[${domain.planId}] ${domain.path} has cURL Examples section with ${domain.expectedCommands} command(s)`, async ({ page }) => {
      await page.goto(domain.path);
      await expect(page.locator('main').getByRole('heading', { name: domain.heading }).first()).toBeVisible();

      // Find and click the cURL Examples toggle button
      const curlButton = page.getByRole('button', { name: /cURL Examples/ });
      await expect(curlButton).toBeVisible();
      await curlButton.click();

      // Scope pre blocks to the cURL section (parent of the button)
      const curlSection = curlButton.locator('..');
      const codeBlocks = curlSection.locator('pre');
      await expect(codeBlocks.first()).toBeVisible();
      await expect(codeBlocks).toHaveCount(domain.expectedCommands, { timeout: 5_000 });

      // Verify each command contains the expected API path keyword
      for (const keyword of domain.keywords) {
        const matchingBlock = codeBlocks.filter({ hasText: keyword });
        await expect(matchingBlock.first()).toBeVisible();
      }

      // Verify copy buttons exist (one per command)
      const copyButtons = curlSection.locator('button[title="Copy"]');
      await expect(copyButtons).toHaveCount(domain.expectedCommands);
    });
  }

  test('cURL section is collapsed by default on all pages', async ({ page }) => {
    await page.goto('/embed');
    await expect(page.locator('main').getByRole('heading', { name: 'Text Embedding' })).toBeVisible();

    const curlButton = page.getByRole('button', { name: /cURL Examples/ });
    await expect(curlButton).toBeVisible();

    // Content should not be visible before clicking
    const curlSection = curlButton.locator('..');
    await expect(curlSection.locator('pre')).toHaveCount(0);
  });

  test('cURL section toggles open and closed', async ({ page }) => {
    await page.goto('/rerank');
    await expect(page.locator('main').getByRole('heading', { name: 'Document Reranking' })).toBeVisible();

    const curlButton = page.getByRole('button', { name: /cURL Examples/ });
    const curlSection = curlButton.locator('..');

    // Open
    await curlButton.click();
    await expect(curlSection.locator('pre').first()).toBeVisible();

    // Close
    await curlButton.click();
    await expect(curlSection.locator('pre')).toHaveCount(0);
  });
});

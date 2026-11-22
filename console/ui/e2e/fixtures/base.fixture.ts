import { test as base } from '@playwright/test';

/**
 * Extended test fixture with helpers for LMSupply Console E2E tests.
 */
export const test = base.extend<{
  /** Wait for initial page load (API calls settle) */
  waitForPageReady: () => Promise<void>;
}>({
  waitForPageReady: async ({ page }, provide) => {
    await provide(async () => {
      // Wait for any loading spinners to disappear
      const spinner = page.locator('.animate-spin');
      if (await spinner.count() > 0) {
        await spinner.first().waitFor({ state: 'hidden', timeout: 15_000 }).catch(() => {
          // Spinner may have already disappeared
        });
      }
    });
  },
});

export { expect } from '@playwright/test';

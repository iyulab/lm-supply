import { test, expect } from './fixtures/base.fixture';

// ================================================================
// Dashboard polling pauses when tab is hidden
// Covers: tab visibility polling pause/resume
// ================================================================

test.describe('Dashboard — tab visibility polling', () => {
  // [2-12] Polling pauses when tab is hidden, resumes when tab is visible
  test('[2-12] dashboard polling pauses on tab hidden and resumes on visible', async ({ page }) => {
    // Track API calls to /api/system/status
    const statusCalls: number[] = [];
    await page.route('**/api/system/status', async (route) => {
      statusCalls.push(Date.now());
      await route.continue();
    });

    await page.goto('/');
    await expect(page.getByRole('heading', { name: /Dashboard/ })).toBeVisible();

    // Wait for initial fetch + at least one poll cycle (5s interval)
    await page.waitForTimeout(6000);
    const callsBeforeHide = statusCalls.length;
    expect(callsBeforeHide).toBeGreaterThanOrEqual(2); // initial + at least 1 poll

    // Simulate tab becoming hidden
    await page.evaluate(() => {
      Object.defineProperty(document, 'hidden', { value: true, writable: true });
      document.dispatchEvent(new Event('visibilitychange'));
    });

    // Wait for what would be 2 poll cycles
    const callsAtHide = statusCalls.length;
    await page.waitForTimeout(11000);
    const callsDuringHidden = statusCalls.length - callsAtHide;

    // Should have 0 calls while hidden (polling stopped)
    expect(callsDuringHidden).toBe(0);

    // Simulate tab becoming visible again
    await page.evaluate(() => {
      Object.defineProperty(document, 'hidden', { value: false, writable: true });
      document.dispatchEvent(new Event('visibilitychange'));
    });

    // Wait for immediate fetch + a poll
    await page.waitForTimeout(6000);
    const callsAfterResume = statusCalls.length - callsAtHide - callsDuringHidden;

    // Should have resumed polling (at least 1 immediate + 1 poll)
    expect(callsAfterResume).toBeGreaterThanOrEqual(2);
  });
});

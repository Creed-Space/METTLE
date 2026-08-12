const { test, expect } = require('@playwright/test');
const AxeBuilder = require('@axe-core/playwright').default;

const pages = ['/', '/test', '/guide', '/about'];
const viewports = [
  { name: 'desktop', width: 1440, height: 900 },
  { name: 'tablet', width: 768, height: 1024 },
  { name: 'mobile', width: 480, height: 900 },
];

for (const viewport of viewports) {
  for (const route of pages) {
    test(`${route} has no serious accessibility violations at ${viewport.name}`, async ({ page }) => {
      await page.setViewportSize(viewport);
      const consoleErrors = [];
      page.on('console', message => {
        if (
          message.type() === 'error' &&
          !message.text().includes('net::ERR_CONNECTION_CLOSED')
        ) {
          consoleErrors.push(message.text());
        }
      });
      await page.goto(route, { waitUntil: 'domcontentloaded' });
      await page.locator('main').waitFor();
      await page.waitForTimeout(250);

      const results = await new AxeBuilder({ page })
        .withTags(['wcag2a', 'wcag2aa', 'wcag21aa', 'wcag22aa'])
        .analyze();
      const materialViolations = results.violations.filter(violation =>
        ['serious', 'critical'].includes(violation.impact),
      );

      expect(materialViolations, JSON.stringify(materialViolations, null, 2)).toEqual([]);
      expect(consoleErrors).toEqual([]);
    });
  }
}

test('keyboard users can skip navigation and operate the mobile menu', async ({ page }) => {
  await page.setViewportSize({ width: 480, height: 900 });
  await page.goto('/');

  await page.keyboard.press('Tab');
  await expect(page.getByRole('link', { name: 'Skip to main content' })).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page.locator('#main-content')).toBeFocused();

  const menu = page.locator('#mobile-menu-btn');
  await menu.focus();
  await page.keyboard.press('Enter');
  await expect(menu).toHaveAttribute('aria-expanded', 'true');
  await expect(page.getByRole('navigation', { name: 'Mobile navigation' })).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(menu).toBeFocused();
  await expect(menu).toHaveAttribute('aria-expanded', 'false');
});

test('reduced motion exposes content without animation loops', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('/', { waitUntil: 'domcontentloaded' });
  await expect(page.locator('.scroll-reveal').first()).toHaveCSS('opacity', '1');
  const typewriter = page.locator('.typewriter-text');
  const initial = await typewriter.textContent();
  await page.waitForTimeout(1_300);
  await expect(typewriter).toHaveText(initial);
});

test('challenge flow announces progress and focuses the final result', async ({ page }) => {
  await page.goto('/test', { waitUntil: 'domcontentloaded' });
  await page.getByLabel('Entity ID (optional)').fill('browser-acceptance-agent');
  await page.getByRole('button', { name: 'Start Verification' }).click();

  const progress = page.getByRole('progressbar', { name: 'Challenge progress' });
  await expect(progress).toHaveAttribute('aria-valuenow', '0');
  for (let completed = 1; completed <= 3; completed += 1) {
    await page.getByRole('button', { name: 'Submit Answer' }).click();
    if (completed < 3) {
      await expect(progress).toHaveAttribute('aria-valuenow', String(completed));
    }
  }

  await expect(page.locator('#result-screen')).toHaveClass(/active/);
  await expect(page.locator('#result-title')).toBeFocused();
  await expect(page.locator('#stat-total')).toHaveText('3');

  await page.getByRole('button', { name: 'Try Again' }).click();
  await expect(page.locator('#start-screen')).toHaveClass(/active/);
  await expect(page.getByLabel('Entity ID (optional)')).toBeFocused();
});

test('API errors are announced and recovery returns to the start', async ({ page }) => {
  await page.route('**/api/session/start', route =>
    route.fulfill({
      status: 503,
      contentType: 'application/json',
      body: JSON.stringify({ detail: 'Temporary test dependency failure' }),
    }),
  );
  await page.goto('/test', { waitUntil: 'domcontentloaded' });
  await page.getByRole('button', { name: 'Start Verification' }).click();

  await expect(page.locator('#error-screen')).toHaveClass(/active/);
  await expect(page.locator('#error-title')).toBeFocused();
  await expect(page.getByRole('alert')).toContainText('Temporary test dependency failure');
  await page.getByRole('button', { name: 'Start Over' }).click();
  await expect(page.locator('#start-screen')).toHaveClass(/active/);
});

test('media has a poster, captions, transcript, and intent-gated payload', async ({ page }) => {
  const videoRequests = [];
  page.on('request', request => {
    if (request.url().includes('mettle-explainer.mp4')) videoRequests.push(request);
  });
  await page.goto('/', { waitUntil: 'domcontentloaded' });

  const video = page.locator('#explainer-video');
  const source = video.locator('source');
  await expect(video).toHaveAttribute('poster', /mettle-explainer-poster\.webp\?v=/);
  await expect(video.locator('track[kind="captions"]')).toHaveAttribute('src', /\.vtt\?v=/);
  await expect(source).not.toHaveAttribute('src', /.+/);
  await expect(source).toHaveAttribute('data-src', /mettle-explainer\.mp4\?v=/);
  await expect(page.getByText('Read the video transcript and assurance note')).toBeVisible();
  await page.waitForTimeout(3000);
  expect(videoRequests).toHaveLength(0);

  const requestStarted = page.waitForRequest(request =>
    request.url().includes('mettle-explainer.mp4'),
  );
  await page.getByRole('button', { name: 'Play the METTLE explainer video' }).click();
  await requestStarted;
  await expect(source).toHaveAttribute('src', /mettle-explainer\.mp4\?v=/);
  expect(videoRequests).toHaveLength(1);
});

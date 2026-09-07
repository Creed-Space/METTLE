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
  await expect(typewriter).toHaveText('a reverse Turing test');
  const initial = await typewriter.textContent();
  await page.waitForTimeout(1_300);
  await expect(typewriter).toHaveText(initial);
});

test('hero entrance fades without moving on load or refresh', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'no-preference' });
  for (const width of [1440, 320]) {
    await page.setViewportSize({ width, height: 900 });
    await page.goto('/');
    for (const refresh of [false, true]) {
      if (refresh) await page.reload();
      await page.evaluate(() => document.fonts.ready);
      const frames = await page.locator('.hero-content > *').evaluateAll(elements =>
        elements.map(element => {
          const animation = element.getAnimations().find(a => a.animationName === 'heroFadeIn');
          if (!animation) throw new Error('Hero entrance animation is missing');
          animation.pause();
          const { delay, duration } = animation.effect.getTiming();
          return [0, 0.5, 1].map(progress => {
            animation.currentTime = delay + duration * progress;
            const rect = element.getBoundingClientRect();
            return { top: rect.top, left: rect.left, opacity: Number(getComputedStyle(element).opacity) };
          });
        })
      );
      expect(frames).toHaveLength(6);
      for (const [start, middle, end] of frames) {
        expect(start.opacity).toBe(0);
        expect(middle.opacity).toBeGreaterThan(0);
        expect(middle.opacity).toBeLessThan(1);
        expect(end.opacity).toBe(1);
        expect(middle.top).toBeCloseTo(start.top, 2);
        expect(end.top).toBeCloseTo(start.top, 2);
        expect(end.left).toBeCloseTo(start.left, 2);
      }
    }
  }
});

test('challenge flow announces progress and focuses the final result', async ({ page }) => {
  await page.goto('/test', { waitUntil: 'domcontentloaded' });
  await page.getByLabel('Entity ID (optional)').fill('browser-acceptance-agent');
  await page.getByRole('button', { name: 'Start Verification' }).click();

  await expect(page.locator('#answer-input')).toBeFocused();
  const progress = page.getByRole('progressbar', { name: 'Challenge progress' });
  await expect(progress).toHaveAttribute('aria-valuenow', '0');
  for (let completed = 1; completed <= 3; completed += 1) {
    await page.getByRole('button', { name: 'Submit Answer' }).click();
    if (completed < 3) {
      await expect(progress).toHaveAttribute('aria-valuenow', String(completed));
      await expect(page.locator('#answer-input')).toBeFocused();
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
  await expect(page.getByRole('button', { name: 'Retry result' })).toBeHidden();
  await page.getByRole('button', { name: 'Start Over' }).click();
  await expect(page.locator('#start-screen')).toHaveClass(/active/);
});

test('result recovery retries reads without repeating answers', async ({ page }, testInfo) => {
  let reads = 0;
  let answers = 0;
  page.on('request', request => {
    if (request.url().endsWith('/api/session/answer')) answers++;
  });
  await page.route('**/api/session/*/result', async route => {
    reads++;
    if (reads < 3) {
      await route.fulfill({ status: 503, contentType: 'application/json',
        body: JSON.stringify({ detail: 'Result temporarily unavailable' }) });
    } else {
      await route.continue();
    }
  });
  await page.goto('/test');
  await page.getByRole('button', { name: 'Start Verification' }).click();
  for (let i = 0; i < 3; i++) {
    await page.getByRole('button', { name: 'Submit Answer' }).click();
    if (i < 2) await expect(page.locator('#progress-bar')).toHaveAttribute('aria-valuenow', String(i + 1));
  }
  const retry = page.getByRole('button', { name: 'Retry result' });
  await expect(retry).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('result-retry.png'), animations: 'disabled' });
  await retry.click();
  await expect(retry).toBeEnabled();
  await expect(page.locator('#error-screen')).toHaveClass(/active/);
  await retry.click();
  await expect(page.locator('#result-title')).toBeFocused();
  expect(reads).toBe(3);
  expect(answers).toBe(3);
  await page.getByRole('button', { name: 'Try Again' }).click();
  await expect(retry).toBeHidden();
});

test('answer failures never offer result recovery', async ({ page }) => {
  await page.route('**/api/session/answer', route => route.fulfill({
    status: 503, contentType: 'application/json', body: JSON.stringify({ detail: 'Answer not accepted' }),
  }));
  await page.goto('/test');
  await page.getByRole('button', { name: 'Start Verification' }).click();
  await page.getByRole('button', { name: 'Submit Answer' }).click();
  await expect(page.locator('#error-screen')).toHaveClass(/active/);
  await expect(page.getByRole('button', { name: 'Retry result' })).toBeHidden();
});

for (const verified of [true, false]) {
  test(`result icon renders for verified=${verified}`, async ({ page }, testInfo) => {
    await page.goto('/test');
    // Synthetic result, used only to inspect both presentation states.
    await page.evaluate(verified => displayResult({ verified, tier: 'bronze',
      passed: verified ? 3 : 0, total: 3, pass_rate: verified ? 1 : 0, results: [] }), verified);
    await page.evaluate(() => document.fonts.ready);
    await expect(page.locator('#result-icon i')).toHaveClass(`fa-solid fa-circle-${verified ? 'check' : 'xmark'}`);
    await page.screenshot({ path: testInfo.outputPath(`result-${verified}.png`), animations: 'disabled' });
  });
}

for (const width of [320, 1440]) {
  test(`small polish fits at ${width}px`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width, height: 900 });
    for (const [route, selector, name] of [['/', '.hero-typewriter', 'home'],
      ['/test', '#start-screen', 'test'], ['/guide', '#quickstart', 'quickstart']]) {
      await page.goto(route);
      if (route === '/') {
        await expect(page.locator('.typewriter-text')).toHaveText('a reverse Turing test');
        const copyGap = await page.evaluate(() => {
          const subtitle = document.querySelector('.hero-subtitle').getBoundingClientRect();
          const question = document.querySelector('.hero-question').getBoundingClientRect();
          return question.top - subtitle.bottom;
        });
        expect(copyGap).toBeGreaterThanOrEqual(4);
        expect(copyGap).toBeLessThanOrEqual(16);
      }
      await page.locator(selector).scrollIntoViewIfNeeded();
      await page.evaluate(() => document.fonts.ready);
      const fits = await page.locator(selector).evaluate(e => {
        const r = e.getBoundingClientRect();
        return r.left >= 0 && r.right <= innerWidth && e.scrollWidth <= e.clientWidth;
      });
      expect(fits).toBe(true);
      await page.screenshot({ path: testInfo.outputPath(`${name}-${width}.png`), animations: 'disabled' });
    }
  });
}

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

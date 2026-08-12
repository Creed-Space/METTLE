const { test, expect } = require('@playwright/test');
const fs = require('node:fs');
const path = require('node:path');

const budget = require('../../frontend-performance-budget.json');

test('landing page stays inside the committed lab performance budget', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.addInitScript(() => {
    window.__mettlePerformance = { cls: 0, lcp: 0, lcpDetail: null, inp: 0 };
    new PerformanceObserver(list => {
      for (const entry of list.getEntries()) {
        if (!entry.hadRecentInput) window.__mettlePerformance.cls += entry.value;
      }
    }).observe({ type: 'layout-shift', buffered: true });
    new PerformanceObserver(list => {
      const entries = list.getEntries();
      if (entries.length) {
        const entry = entries[entries.length - 1];
        window.__mettlePerformance.lcp = entry.startTime;
        window.__mettlePerformance.lcpDetail = {
          tagName: entry.element?.tagName || null,
          id: entry.element?.id || null,
          className: entry.element?.className || null,
          text: entry.element?.textContent?.trim().slice(0, 120) || null,
          url: entry.url || null,
          size: entry.size,
          renderTime: entry.renderTime,
          loadTime: entry.loadTime,
        };
      }
    }).observe({ type: 'largest-contentful-paint', buffered: true });
    new PerformanceObserver(list => {
      for (const entry of list.getEntries()) {
        window.__mettlePerformance.inp = Math.max(
          window.__mettlePerformance.inp,
          entry.duration,
        );
      }
    }).observe({ type: 'event', buffered: true, durationThreshold: 16 });
  });

  await page.goto('/', { waitUntil: 'domcontentloaded' });
  await page.locator('main').waitFor();
  await page.waitForTimeout(500);

  const metrics = await page.evaluate(() => {
    const resources = performance.getEntriesByType('resource');
    const video = resources
      .filter(entry => entry.name.includes('mettle-explainer.mp4'))
      .reduce((total, entry) => total + entry.transferSize, 0);
    const critical = resources
      .filter(entry => !entry.name.includes('mettle-explainer.mp4'))
      .reduce((total, entry) => total + entry.transferSize, 0);
    const navigation = performance.getEntriesByType('navigation')[0];
    const slowestResources = resources
      .map(entry => ({
        name: entry.name,
        duration: entry.duration,
        startTime: entry.startTime,
        responseEnd: entry.responseEnd,
        transferSize: entry.transferSize,
      }))
      .sort((left, right) => right.responseEnd - left.responseEnd)
      .slice(0, 10);
    return {
      ...window.__mettlePerformance,
      criticalTransferBytes: critical,
      initialVideoTransferBytes: video,
      navigation: navigation
        ? {
            responseStart: navigation.responseStart,
            domContentLoadedEventEnd: navigation.domContentLoadedEventEnd,
            loadEventEnd: navigation.loadEventEnd,
          }
        : null,
      slowestResources,
      recordedAt: new Date().toISOString(),
      url: location.href,
    };
  });

  const evidencePath = testInfo.outputPath('performance-budget.json');
  fs.mkdirSync(path.dirname(evidencePath), { recursive: true });
  fs.writeFileSync(evidencePath, JSON.stringify({ budget, metrics }, null, 2));
  await testInfo.attach('performance-budget', {
    path: evidencePath,
    contentType: 'application/json',
  });

  expect(metrics.lcp).toBeGreaterThan(0);
  expect(metrics.lcp).toBeLessThanOrEqual(budget.lcp_ms_max);
  expect(metrics.inp).toBeLessThanOrEqual(budget.inp_ms_max);
  expect(metrics.cls).toBeLessThanOrEqual(budget.cls_max);
  expect(metrics.criticalTransferBytes).toBeLessThanOrEqual(
    budget.critical_transfer_bytes_max,
  );
  expect(metrics.initialVideoTransferBytes).toBeLessThanOrEqual(
    budget.initial_video_transfer_bytes_max,
  );
});

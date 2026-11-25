const { test, expect } = require('../browser-fixtures');
const fs = require('node:fs');

test('accepted explainer plays, seeks and loads measured captions', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto('/', { waitUntil: 'domcontentloaded' });
  await page.getByRole('button', { name: 'Play the METTLE explainer video' }).click();
  const video = page.locator('#explainer-video');
  await expect.poll(() => video.evaluate(v => v.readyState)).toBeGreaterThanOrEqual(2);
  const metadata = await video.evaluate(v => ({
    duration: v.duration, width: v.videoWidth, height: v.videoHeight,
  }));
  expect(metadata.duration).toBeCloseTo(371.066667, 1);
  expect(metadata.width).toBe(1920);
  expect(metadata.height).toBe(1080);
  const before = await video.evaluate(v => v.currentTime);
  await expect.poll(() => video.evaluate(v => v.currentTime)).toBeGreaterThan(before + 0.25);
  await video.evaluate(v => {
    v.pause();
    v.textTracks[0].mode = 'showing';
  });
  await expect.poll(() => video.evaluate(v => v.textTracks[0].cues?.length || 0)).toBe(125);
  const seeks = [];
  for (const time of [121.1, 363]) {
    await video.evaluate((v, seconds) => new Promise((resolve, reject) => {
      v.addEventListener('seeked', resolve, { once: true });
      v.addEventListener('error', () => reject(new Error('Media seek failed')), { once: true });
      v.currentTime = seconds;
    }), time);
    await expect.poll(() => video.evaluate(v => v.textTracks[0].activeCues?.length || 0))
      .toBeGreaterThan(0);
    seeks.push(await video.evaluate(v => ({
      time: v.currentTime, cues: Array.from(v.textTracks[0].activeCues, cue => cue.text),
      error: v.error?.code || null,
    })));
    await video.screenshot({ path: testInfo.outputPath(`video-${time}.png`) });
  }
  await page.setViewportSize({ width: 480, height: 900 });
  await video.scrollIntoViewIfNeeded();
  expect(await video.evaluate(v => v.getBoundingClientRect().right <= innerWidth)).toBe(true);
  await video.screenshot({ path: testInfo.outputPath('video-mobile.png') });
  await video.evaluate(v => v.play());
  await page.mouse.move(0, 0);
  await expect.poll(() => video.evaluate(v => v.currentTime)).toBeGreaterThan(367.8);
  await video.screenshot({ path: testInfo.outputPath('video-mobile-playing.png') });
  await video.evaluate(v => v.pause());
  const evidence = { metadata, playbackAdvanced: true, captions: 125, seeks, mediaError: null };
  fs.writeFileSync(testInfo.outputPath('video-playback.json'), JSON.stringify(evidence, null, 2));
  await testInfo.attach('video-playback', {
    path: testInfo.outputPath('video-playback.json'), contentType: 'application/json',
  });
});

const { defineConfig } = require('@playwright/test');

const PORT = 8765;

module.exports = defineConfig({
  testDir: './tests/browser',
  outputDir: 'test-results/browser-artifacts',
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  workers: process.env.CI ? 2 : 2,
  reporter: process.env.CI
    ? [['line'], ['html', { outputFolder: 'playwright-report', open: 'never' }]]
    : 'line',
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    browserName: 'chromium',
    colorScheme: 'dark',
    reducedMotion: 'reduce',
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
    video: 'retain-on-failure',
  },
  timeout: 60_000,
  webServer: {
    command: [
      'METTLE_ENVIRONMENT=development',
      'METTLE_DEV_MODE=true',
      'METTLE_SECRET_KEY=browser-test-secret-key-12345',
      'METTLE_ADMIN_API_KEY=browser-test-admin-key-12345', // pragma: allowlist secret, public test fixture
      `python3 -m uvicorn main:app --host 127.0.0.1 --port ${PORT}`,
    ].join(' '),
    url: `http://127.0.0.1:${PORT}/api/health/ready`,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
    stdout: 'pipe',
    stderr: 'pipe',
  },
});

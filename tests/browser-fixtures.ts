import { test as base } from "@playwright/test";
export * from "@playwright/test";

// Derive from the resolved engine, including projects created by --browser=all.
export const test = base.extend({
  launchOptions: [
    async ({ browserName, launchOptions }, use) => {
      await use(
        browserName === "chromium"
          ? { ...launchOptions, channel: "chrome", chromiumSandbox: true }
          : launchOptions,
      );
    },
    { scope: "worker" },
  ],
});

export default test;

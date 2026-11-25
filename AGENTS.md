# Agent instructions

## Browser test defaults

Browser specs import `test` from `tests/browser-fixtures.ts`. It derives sandboxed stable Chrome from the resolved Chromium engine and preserves Firefox and WebKit. Use the normal project test commands; security flags belong in the shared fixture.

Working if default and `--browser=all` runs resolve Chrome with sandboxing only for Chromium, and maintained browser specs do not import the unconfigured base test directly.

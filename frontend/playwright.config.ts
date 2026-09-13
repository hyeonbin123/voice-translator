import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './e2e',
  outputDir: '../work/round11-browser',
  fullyParallel: false,
  workers: 1,
  reporter: 'list',
  use: {
    channel: 'msedge',
    baseURL: 'http://127.0.0.1:15174',
    viewport: { width: 360, height: 800 },
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
  },
  webServer: {
    command: 'npm run dev -- --host 127.0.0.1 --port 15174 --strictPort',
    url: 'http://127.0.0.1:15174',
    reuseExistingServer: false,
    env: { VITE_TRANSLATION_MOCK: 'false' },
  },
})

import react from '@vitejs/plugin-react'
import { readFileSync } from 'node:fs'
import { defineConfig } from 'vitest/config'

export default defineConfig({
  plugins: [react(), {
    name: 'conversation-asset-licenses',
    generateBundle() {
      for (const name of ['SILERO-LICENSE.txt', 'ONNXRUNTIME-LICENSE.txt']) {
        this.emitFile({ type: 'asset', fileName: `assets/${name}`,
          source: readFileSync(new URL(`./src/conversation/assets/${name}`, import.meta.url), 'utf8') })
      }
    },
  }],
  resolve: { conditions: ['onnxruntime-web-use-extern-wasm', 'module', 'browser', 'development|production'] },
  server: {
    // The API runs separately (uvicorn on :8000) and every API route is under /api.
    proxy: { '/api': { target: 'http://localhost:8000', ws: true } },
  },
  test: {
    include: ['src/**/*.test.{ts,tsx}'],
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
  },
})

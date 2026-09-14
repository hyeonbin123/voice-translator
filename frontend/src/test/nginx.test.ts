// @vitest-environment node
import { expect, it } from 'vitest'
import configSource from '../../nginx.conf?raw'
import viteSource from '../../vite.config.ts?raw'

const config = configSource.replace(/#.*$/gm, '')

it('adds JavaScript modules alongside the full image MIME table at server scope', () => {
  // Restrict the prefix to directives: a nested include would not preserve the table.
  expect(config).toMatch(/^\s*server\s*\{[^{}]*include\s+\/etc\/nginx\/mime\.types\s*;\s*types\s*\{\s*(?:application|text)\/javascript\s+mjs\s*;\s*\}/)
  // A location-level types block would discard the server table again.
  expect(config.match(/\btypes\s*\{/g)).toHaveLength(1)
  expect(config).not.toMatch(/\bdefault_type\b/)
})

it('serves missing runtime assets as 404 instead of the SPA HTML fallback', () => {
  expect(config).toMatch(/location\s+\/assets\/\s*\{\s*try_files\s+\$uri\s+=404\s*;\s*\}/)
})

it('upgrades only the live endpoint and preserves the ordinary API forwarding contract', () => {
  const live = config.match(/location\s*=\s*\/api\/translate\/live\s*\{([^}]+)\}/)?.[1] ?? ''
  const http = config.match(/location\s+\/api\/\s*\{([^}]+)\}/)?.[1] ?? ''
  expect(live).toMatch(/proxy_http_version\s+1\.1;/)
  expect(live).toMatch(/proxy_set_header\s+Upgrade\s+\$http_upgrade;/)
  expect(live).toMatch(/proxy_set_header\s+Connection\s+"upgrade";/)
  expect(live).toMatch(/proxy_read_timeout\s+3600s;/)
  for (const block of [live, http]) {
    expect(block).toMatch(/set\s+\$api_upstream\s+api:8000;/)
    expect(block).toMatch(/proxy_pass\s+http:\/\/\$api_upstream\$request_uri;/)
    for (const [header, value] of [['Host', '$host'], ['X-Real-IP', '$remote_addr'],
      ['X-Forwarded-For', '$proxy_add_x_forwarded_for'], ['X-Forwarded-Proto', '$scheme']]) {
      expect(block).toContain(`proxy_set_header ${header} ${value};`)
    }
  }
  expect(http).toContain('proxy_read_timeout 300s;')
  expect(http).not.toContain('Upgrade')
  expect(config).toContain('client_max_body_size 11m;')
  expect(viteSource).toMatch(/'\/api':\s*\{\s*target:\s*'http:\/\/localhost:8000',\s*ws:\s*true\s*\}/)
})

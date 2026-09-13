// @vitest-environment node
import { expect, it } from 'vitest'
import configSource from '../../nginx.conf?raw'

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

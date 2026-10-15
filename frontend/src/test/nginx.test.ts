// @vitest-environment node
// [파일 역할] 배포용 nginx 설정(frontend/nginx.conf)과 개발 서버 프록시(vite.config.ts)를 글자로 읽어, 고장
// 났던 적이 있거나 계약으로 정한 줄이 그대로 있는지 확인하는 테스트 묶음.
// [왜 필요한가] nginx 설정은 Docker 이미지로 띄워야만 동작을 볼 수 있어 실수가 늦게 드러난다. 예를 들어
// .mjs가 application/octet-stream으로 나가 대화 모드가 배포 서비스에서만 준비되지 않은 문제가 있었다(397732f,
// Vite 개발 서버에서는 드러나지 않음). 그런 줄을 일반 단위 테스트로 묶어 두면 npm test(CI 포함)에서 바로
// 잡힌다.
// 맨 첫 줄의 주석은 이 파일만 jsdom 대신 node 환경에서 돌게 한다(DOM이 필요 없음). ?raw 붙은 import는 Vite가
// 파일 내용을 문자열로 넘겨 준다.

import { expect, it } from 'vitest'
import configSource from '../../nginx.conf?raw'
import viteSource from '../../vite.config.ts?raw'
import microphoneSource from '../conversation/microphone.ts?raw'

// nginx 주석(# 뒤)을 먼저 지운다. 주석 속 단어가 정규식에 걸려 잘못 통과하는 것을 막는다.
const config = configSource.replace(/#.*$/gm, '')

// 확인: server 블록 바로 안에서 nginx 기본 MIME 표(mime.types)를 include한 뒤 types 블록에 mjs를
// 자바스크립트로 더하는지.
// 왜: onnxruntime-web의 WASM 로더가 .mjs 파일인데 nginx 기본 표에는 mjs가 없다. 브라우저는 자바스크립트가
// 아닌 MIME으로 온 모듈을 거부하므로 대화 모드가 준비되지 않는다. 또 server 단계에 types 블록을 쓰면 물려받던
// 표 전체가 그 블록으로 바뀌므로 include를 같은 단계에 함께 둬야 .js·.css·.wasm 등도 계속 올바른 형식으로
// 나간다. types 블록이 하나뿐인지, default_type을 바꾸지 않았는지도 함께 본다.
it('adds JavaScript modules alongside the full image MIME table at server scope', () => {
  // 앞부분은 지시어만 허용한다: 블록 안에 들어간 include로는 표가 유지되지 않는다.
  expect(config).toMatch(/^\s*server\s*\{[^{}]*include\s+\/etc\/nginx\/mime\.types\s*;\s*types\s*\{\s*(?:application|text)\/javascript\s+mjs\s*;\s*\}/)
  // location 단계의 types 블록이 있으면 server 단계의 표를 다시 버리게 된다.
  expect(config.match(/\btypes\s*\{/g)).toHaveLength(1)
  expect(config).not.toMatch(/\bdefault_type\b/)
})

// 확인: /assets/ 아래에 없는 파일은 index.html 대신 404를 돌려주는지, 해시 붙은 자산은 1년 immutable로
// 캐시하고 해시 없는 라이선스 원문과 index.html은 매번 확인(no-cache)하는지.
// 왜: 화면 경로용 대체(try_files ... /index.html)가 자산에도 적용되면, 없는 WASM·모델 파일 요청에 HTML이
// 200으로 와서 원인을 알기 어려운 오류가 난다. 404면 바로 드러난다. 해시 붙은 파일은 내용이 바뀌면 이름이
// 바뀌므로 오래 캐시해도 안전하고, 대화 모드의 14MB WASM을 다시 받지 않게 한다. 이름이 고정된 파일을 오래
// 캐시하면 배포 뒤에도 옛 내용이 남는다.
it('serves hashed assets as long-lived 404-or-file and keeps fixed names fresh', () => {
  const assets = config.match(/location\s+\/assets\/\s*\{([^}]+)\}/)?.[1] ?? ''
  expect(assets).toMatch(/try_files\s+\$uri\s+=404\s*;/)
  expect(assets).toMatch(/add_header\s+Cache-Control\s+"public, max-age=31536000, immutable"\s+always\s*;/)
  const licenses = config.match(/location\s+~\s+\^\/assets\/\[A-Z\]\+-LICENSE\\\.txt\$\s*\{([^}]+)\}/)?.[1] ?? ''
  expect(licenses).toMatch(/try_files\s+\$uri\s+=404\s*;/)
  expect(licenses).toMatch(/add_header\s+Cache-Control\s+"no-cache"\s+always\s*;/)
  for (const name of ['SILERO-LICENSE.txt', 'ONNXRUNTIME-LICENSE.txt']) {
    expect(/^\/assets\/[A-Z]+-LICENSE\.txt$/.test(`/assets/${name}`)).toBe(true)
  }
  expect(/^\/assets\/[A-Z]+-LICENSE\.txt$/.test('/assets/index-PxBfrqvz.js')).toBe(false)
  const spa = config.match(/location\s+\/\s*\{([^}]+)\}/)?.[1] ?? ''
  expect(spa).toMatch(/try_files\s+\$uri\s+\$uri\/\s+\/index\.html\s*;/)
  expect(spa).toMatch(/add_header\s+Cache-Control\s+"no-cache"\s+always\s*;/)
})

// 확인: server 단계의 보안 머리글이 모든 응답(always)에 붙고, 캐시 머리글을 가진 location에서도 사라지지
// 않게 add_header_inherit merge가 있는지. CSP는 같은 출처만 허용하고, WebAssembly 컴파일만 따로 허용하며,
// 인라인 스크립트·스타일과 data: 스크립트를 허용하지 않는지.
// 왜: nginx는 location에 add_header가 하나라도 있으면 server 단계의 add_header를 모두 버린다(merge 없이). 그러면
// 화면 HTML(location /)과 자산에서 CSP가 빠진다. CSP는 XSS가 나도 끼워 넣은 스크립트가 돌지 않게 하는 장치다
// (refresh 토큰이 sessionStorage에 있어 특히 중요, docs/security.md). 'unsafe-inline'·'unsafe-eval'·data:를
// 넣으면 그 효과가 사라진다.
it('sends the security headers on every response, merged into locations that add their own', () => {
  // location 블록을 지운 server 단계만 본다. \b와 공백은 Permissions-Policy의 'geolocation' 같은 단어 속
  // 'location'을 블록으로 잘못 보지 않게 한다.
  const server = config.replace(/\blocation\s[^{]*\{[^}]*\}/g, '')
  expect(server).toMatch(/add_header_inherit\s+merge\s*;/)
  expect(server).toMatch(/server_tokens\s+off\s*;/)
  const header = (name: string) =>
    server.match(new RegExp(String.raw`add_header\s+${name}\s+"([^"]+)"\s+always\s*;`))?.[1]
  const csp = header('Content-Security-Policy') ?? ''
  const directives = Object.fromEntries(csp.split(';').map((part) => part.trim().split(/\s+/)).filter((part) => part[0])
    .map(([name, ...values]) => [name, values]))
  expect(directives['default-src']).toEqual(["'self'"])
  expect(directives['script-src']).toEqual(["'self'", "'wasm-unsafe-eval'"])
  expect(directives['style-src']).toEqual(["'self'"])
  expect(directives['connect-src']).toEqual(["'self'"])
  expect(directives['media-src']).toEqual(["'self'", 'blob:'])
  expect(directives['frame-ancestors']).toEqual(["'none'"])
  expect(directives['object-src']).toEqual(["'none'"])
  expect(directives['base-uri']).toEqual(["'none'"])
  // 'wasm-unsafe-eval'은 허용하므로 따옴표까지 붙여 일반 'unsafe-eval'만 찾는다.
  expect(csp).not.toMatch(/'unsafe-inline'|'unsafe-eval'|data:|\*/)
  expect(header('X-Frame-Options')).toBe('DENY')
  expect(header('X-Content-Type-Options')).toBe('nosniff')
  expect(header('Referrer-Policy')).toBe('no-referrer')
  expect(header('Permissions-Policy')).toMatch(/^microphone=\(self\), camera=\(\)/)
  expect(header('Cross-Origin-Opener-Policy')).toBe('same-origin')
  expect(header('Cross-Origin-Resource-Policy')).toBe('same-origin')
})

// 확인: gzip이 켜져 있고 WASM·자바스크립트·CSS·JSON을 압축하며, 이미 압축된 음성(audio/wav)과 거의 줄지 않는
// 모델 파일(application/octet-stream)은 넣지 않는지.
// 왜: 대화 모드가 처음 받는 onnxruntime-web WASM이 약 14MB인데 gzip이면 약 3.7MB다(README). 압축이 꺼져 있던
// 동안에는 README의 그 수치가 배포 이미지에서 성립하지 않았다.
it('compresses the runtime assets but not audio or the model', () => {
  expect(config).toMatch(/\bgzip\s+on\s*;/)
  const types = config.match(/gzip_types\s+([^;]+);/)?.[1].split(/\s+/) ?? []
  for (const type of ['application/wasm', 'application/javascript', 'text/css', 'application/json']) {
    expect(types).toContain(type)
  }
  expect(types).not.toContain('application/octet-stream')
  expect(types.some((type) => type.startsWith('audio/'))).toBe(false)
})

// 확인: 마이크 워클릿을 data: URL로 묶지 않고 별도 파일로 내보내게 import하는지(?url&no-inline).
// 왜: 워클릿이 4KB보다 작아 Vite가 기본으로 data: URL로 묶어 넣었는데, 위 CSP(script-src 'self')가 data:
// 스크립트를 막아 배포 이미지에서 대화 모드의 마이크가 열리지 않게 된다.
it('ships the microphone worklet as a same-origin file the CSP allows', () => {
  expect(microphoneSource).toMatch(/import\s+workletUrl\s+from\s+'\.\/capture\.worklet\.js\?url&no-inline'/)
})

// 확인: 웹소켓 업그레이드(HTTP/1.1, Upgrade·Connection 헤더, 1시간 읽기 제한)는 /api/translate/live 한
// 경로에만 있고, 일반 /api/ 경로는 업그레이드 헤더 없이 300초 제한으로 넘기는지. 두 경로 모두 요청할 때마다
// api 서비스 이름을 다시 찾는 변수 방식 proxy_pass와 전달 헤더 4개를 유지하는지, 업로드 한도 11m, 그리고 개발
// 서버 프록시가 웹소켓(ws: true)을 넘기는지.
// 왜: 동시통역과 로그인한 대화 모드는 웹소켓 연결 하나를 길게 쓰므로 업그레이드와 긴 읽기 제한이
// 필요하다(c005648). 변수 방식은 api 컨테이너만 다시 만들었을 때 nginx가 옛 주소로 보내 502가 나던 문제를
// 고친 것이다(fb50252). 11m은 API의 10MB 파일 한도에 multipart 머리글 여유를 더한 값이다(8641f0c).
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

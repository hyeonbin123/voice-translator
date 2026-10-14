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

// 확인: /assets/ 아래에 없는 파일은 index.html 대신 404를 돌려주는지.
// 왜: 화면 경로용 대체(try_files ... /index.html)가 자산에도 적용되면, 없는 WASM·모델 파일 요청에 HTML이
// 200으로 와서 원인을 알기 어려운 오류가 난다. 404면 바로 드러난다.
it('serves missing runtime assets as 404 instead of the SPA HTML fallback', () => {
  expect(config).toMatch(/location\s+\/assets\/\s*\{\s*try_files\s+\$uri\s+=404\s*;\s*\}/)
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

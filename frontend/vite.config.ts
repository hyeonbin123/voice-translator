// [파일 역할] 프런트엔드 빌드 도구 Vite와 단위 테스트 도구 Vitest의 설정을 한 파일에 둔다. `npm run dev`·`npm
// run build`·`npm test`가 모두 이 파일을 읽는다.
// [주요 구성] (1) React 플러그인과, 대화 모드용 라이선스 파일을 배포본에 넣는 작은 플러그인 (2)
// onnxruntime-web이 WASM을 별도 파일로 받도록 고르는 해석 조건 (3) 개발 중 /api 요청·웹소켓을 백엔드(:8000)로
// 넘기는 프록시 (4) Vitest 설정.
// [연결] src/test/nginx.test.ts가 이 파일의 프록시 줄을 글자 그대로 읽어, 배포용 nginx.conf와 같은
// 경로(웹소켓 포함)를 넘기는지 확인한다.

// react: JSX 변환과 개발 중 빠른 새로 고침. readFileSync: 빌드 때 라이선스 원문을 읽는다. defineConfig를
// vitest/config에서 가져오는 이유는 아래 test 항목의 타입까지 함께 검사받기 위해서다.
import react from '@vitejs/plugin-react'
import { readFileSync } from 'node:fs'
import { defineConfig } from 'vitest/config'

export default defineConfig({
  // 두 번째 플러그인(conversation-asset-licenses): 대화 모드는 Silero VAD 모델 파일과 onnxruntime-web의
  // WASM을 화면과 함께 배포한다. 둘 다 MIT라 라이선스 원문을 함께 내보내야 하므로(README 라이선스 표
  // '라이선스 원문 동봉'), 빌드 결과의 assets/ 폴더에 두 원문을 넣는다.
  plugins: [react(), {
    name: 'conversation-asset-licenses',
    // generateBundle은 빌드(번들 쓰기 직전)에만 불리는 훅이라 개발 서버에서는 돌지 않는다. emitFile에
    // fileName을 직접 주면 해시가 붙지 않아 배포본에서 assets/SILERO-LICENSE.txt처럼 늘 같은 주소로 찾을 수
    // 있다. 원문은 src/conversation/assets/에 두고 읽는다.
    generateBundle() {
      for (const name of ['SILERO-LICENSE.txt', 'ONNXRUNTIME-LICENSE.txt']) {
        this.emitFile({ type: 'asset', fileName: `assets/${name}`,
          source: readFileSync(new URL(`./src/conversation/assets/${name}`, import.meta.url), 'utf8') })
      }
    },
  }],
  // onnxruntime-web 패키지는 import 조건에 따라 WASM을 자바스크립트 안에 묻은 판(기본)과 WASM을 따로 받는
  // 판(onnxruntime-web-use-extern-wasm 조건)을 내보낸다. 이 조건을 넣어 따로 받는 판을 고르고,
  // sileroRuntime.ts가 ?url로 가져온 로컬 WASM 주소를 env.wasm.wasmPaths로 넘긴다. 그래서 실행 중에 CDN이나
  // 외부 모델 서버에 접속하지 않는다(src/conversation/assets/SOURCE.md).
  // 나머지 세 값(module, browser, development|production)은 Vite의 기본 조건이다. conditions를 직접 주면
  // 기본값을 대신하므로 다시 적어 둔다.
  resolve: { conditions: ['onnxruntime-web-use-extern-wasm', 'module', 'browser', 'development|production'] },
  server: {
    // API 서버는 따로 돈다(uvicorn, :8000). 모든 API 경로는 /api 아래에 있다.
    // ws: true는 웹소켓 업그레이드 요청도 넘기게 한다. 동시통역과 로그인한 대화 모드·두 사람 대화가 WS
    // /api/translate/live를 쓰기 때문이다(c005648에서 추가). nginx.test.ts가 이 한 줄의 모양을 정규식으로
    // 확인하므로 줄을 나누면 그 테스트가 실패한다.
    proxy: { '/api': { target: 'http://localhost:8000', ws: true } },
  },
  // Vitest 설정.
  // - include: src 아래의 .test.ts·.test.tsx만 모은다. e2e/의 Playwright 검사는 브라우저가 필요해 여기서 돌지
  //   않게 한다(faf1e55)
  // - environment jsdom: 브라우저 DOM을 흉내 내는 환경에서 React 화면을 그려 본다. DOM이 필요 없는 파일은
  //   파일 맨 위 주석으로 node 환경을 따로 고른다(nginx.test.ts)
  // - setupFiles: 테스트 파일마다 먼저 실행해 jest-dom 비교 함수를 등록하고 테스트 뒤 화면을
  //   치운다(src/test/setup.ts)
  test: {
    include: ['src/**/*.test.{ts,tsx}'],
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
  },
})

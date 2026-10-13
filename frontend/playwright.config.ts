// [파일 역할] `npm run test:a11y`(실제 브라우저 접근성 검사)의 Playwright 설정이다.
// e2e/accessibility.spec.ts를 Microsoft Edge로 돌려 모든 조작이 Tab·Enter 키만으로 되는지, axe 검사에서 WCAG
// 2.2 AA 위반이 없는지, 360px 폭에서 가로로 넘치지 않는지를 화면마다 확인한다(faf1e55).
// [왜 따로 두나] Vitest 단위 테스트(jsdom)는 실제 레이아웃·포커스 테두리·색 대비를 계산하지 못한다. 그래서
// 진짜 브라우저 검사를 따로 둔다. README 테스트 절에 적힌 대로 이 검사는 CI에서 돌리지 않고 로컬에서만
// 돌린다.
// [흐름] Vite 개발 서버를 전용 포트로 띄운다 → 검사 파일이 API·마이크·음성 합성 같은 경계만 가짜로 바꾼다 →
// 화면을 키보드로 조작하며 검사한다 → 화면마다 스크린샷을 남긴다.

import { defineConfig } from '@playwright/test'

export default defineConfig({
  // 브라우저 검사 파일 폴더. src 아래의 Vitest 테스트와 섞이지 않게 따로 둔다(vite.config.ts의 test.include가
  // src만 모은다).
  testDir: './e2e',
  // 실패 스크린샷, trace, 검사 중 audit()이 찍는 화면 PNG가 저장되는 곳. git이 추적하지 않는 프로젝트 루트의
  // work/ 폴더(.gitignore)에 두어 결과물이 커밋에 섞이지 않게 한다. README의 화면 그림도 이 검사가 찍은
  // 것이다.
  outputDir: '../work/round11-browser',
  // 개발 서버 하나를 띄워 놓고 검사를 작업자 1개로 하나씩 순서대로 돌린다.
  fullyParallel: false,
  workers: 1,
  reporter: 'list',
  use: {
    // Playwright가 따로 받는 Chromium 대신 PC에 설치된 Microsoft Edge를 쓴다(README: Playwright + Microsoft
    // Edge).
    channel: 'msedge',
    // 평소 개발 서버 포트(5173)와 다른 전용 포트. 손으로 띄워 둔 개발 서버와 부딪히지 않는다.
    baseURL: 'http://127.0.0.1:15174',
    // 가장 좁은 휴대폰 폭(360px)을 기준으로 검사한다. 넓은 화면에서 되는 배치가 좁은 화면에서 깨지는 것을
    // 잡으려는 것이다.
    viewport: { width: 360, height: 800 },
    // 자동 스크린샷과 trace(조작 기록)는 실패한 검사에만 남겨 원인을 볼 수 있게 한다.
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
  },
  // 검사 전에 Playwright가 개발 서버를 직접 띄우고 끝나면 내린다.
  // - strictPort: 포트가 차 있으면 다른 포트로 옮기지 않고 실패한다. 옮기면 baseURL과 어긋나기 때문이다
  // - reuseExistingServer: false: 이미 떠 있는 서버를 재사용하지 않아 다른 설정으로 뜬 서버를 검사하는 일을
  //   막는다
  // - VITE_TRANSLATION_MOCK=false: 예시(가짜 번역) 모드를 끄고 실제 API 호출 코드를 탄다. API 응답은 검사
  //   파일이 page.route로 가짜 값을 돌려준다
  webServer: {
    command: 'npm run dev -- --host 127.0.0.1 --port 15174 --strictPort',
    url: 'http://127.0.0.1:15174',
    reuseExistingServer: false,
    env: { VITE_TRANSLATION_MOCK: 'false' },
  },
})

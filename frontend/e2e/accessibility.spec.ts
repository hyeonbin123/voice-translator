// [파일 역할] `npm run test:a11y`가 돌리는 실제 브라우저(Microsoft Edge) 접근성 검사다(설정은
// playwright.config.ts). 화면마다 모든 조작을 Tab·Enter·방향키·Space 같은 키보드만으로 하고, axe 검사(WCAG
// 2.2 AA까지)에 위반이 없는지, 360px 폭에서 가로로 넘치지 않는지 확인하며 스크린샷을 남긴다(README 테스트 절,
// README의 화면 그림도 이 검사가 찍은 것).
// [왜 따로 두나] Vitest 단위 테스트(jsdom)는 실제 포커스 테두리·레이아웃·색 대비를 계산하지 못한다. 그래서
// 진짜 브라우저 검사를 로컬에서만 따로 돌린다(CI에서는 돌리지 않음).
// [방법] 화면과 API 클라이언트 코드는 실제로 돌리고, 경계만 가짜로 바꾼다: 서버 API(page.route),
// 웹소켓(page.routeWebSocket), 마이크·녹음기·오디오 처리, 브라우저 음성 합성, 말소리 판정 모델(ONNX) 실행기.
// [구성] 대화 모드, 두 사람 대화, 동시통역, 회원가입·로그인·로그아웃, 글자 번역, 녹음·파일 업로드, 기록
// 목록·상세·삭제, 오류와 다시 시도.
// [참고] 로그인한 대화 모드·두 사람 대화는 커밋 00bf7d7부터 웹소켓 미리 처리를 쓴다. 이 파일은 그 뒤에도 두
// 검사의 가짜 응답을 HTTP 업로드 경로에만 걸어 두어, 웹소켓이 개발 서버의 프록시로 나가 연결에 실패하고 두
// 검사가 '듣는 중'을 기다리다 실패했다. 2026-10-13부터 두 검사도 동시통역 검사처럼 가짜 웹소켓 서버로 답하고,
// HTTP 업로드가 한 번도 없었는지 함께 확인한다.

import { test, expect, type Page, type Locator } from '@playwright/test'
// AxeBuilder: 페이지에 axe-core를 넣어 WCAG 규칙 위반을 자동으로 찾는다.
import AxeBuilder from '@axe-core/playwright'

// 가짜 로그인 응답(토큰)과 가짜 번역 결과 하나. 번역 결과는 서버 음성이 없는 상태(tts_error)라 화면이
// 브라우저 내장 음성으로 읽는 경로를 탄다.
const tokens = { access_token: 'access', refresh_token: 'refresh', expires_in: 1800 }
const item = {
  id: 'entry', mode: 'text', source_lang: 'ko', target_lang: 'en', source_text: '안녕하세요',
  translated_text: 'Hello', audio_id: null, tts_error: 'Speech synthesis is not available',
  stt_ms: null, mt_ms: 120, tts_ms: null, created_at: '2026-09-13T00:00:00Z',
}

// tabTo: 대상에 포커스가 갈 때까지 Tab을 최대 90번 누른다. 마우스 클릭이나 focus() 대신 실제 Tab 순서로만
// 닿을 수 있는지 확인하고, 포커스가 닿았을 때 3px 실선 테두리(styles.css의 :focus-visible)가 보이는지도 함께
// 본다. 끝까지 닿지 못하면 실패한다. locator.focus나 click은 절대 쓰지 않는다.
async function tabTo(page: Page, target: Locator) {
  for (let i = 0; i < 90; i++) {
    if (await target.evaluate((el) => el === document.activeElement)) {
      await expect(target).toHaveCSS('outline-style', 'solid')
      await expect(target).toHaveCSS('outline-width', '3px')
      return
    }
    await page.keyboard.press('Tab')
  }
  throw new Error(`Unreachable by Tab: ${await target.textContent()}`)
}

// Tab으로 대상에 간 뒤 Enter로 누른다(키보드만으로 단추·링크 실행).
async function activate(page: Page, target: Locator) {
  await tabTo(page, target)
  await page.keyboard.press('Enter')
}

// 화면 한 상태의 검사: axe(wcag2a·2aa·21aa·22aa 태그) 위반이 0개인지, 문서 폭이 창 폭을 넘지 않는지(가로
// 스크롤 없음) 확인하고, 이름을 붙인 전체 화면 스크린샷을 검사 결과 폴더에 남긴다.
async function audit(page: Page, name: string) {
  const results = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa', 'wcag22aa']).analyze()
  expect(results.violations, `${name}: accessibility`).toEqual([])
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), `${name}: horizontal overflow`).toBe(true)
  await page.screenshot({ path: test.info().outputPath(`${name}.png`), fullPage: true })
}

// 모든 검사의 공통 준비: 브라우저 기능과 서버 API를 가짜로 바꾼다.
test.beforeEach(async ({ page }) => {
  // 페이지가 뜨기 전에 실행할 스크립트. 브라우저 음성 합성(speechSynthesis), 마이크 권한(getUserMedia),
  // 녹음기(MediaRecorder)를 가짜로 바꿔 실제 마이크·스피커 없이 녹음과 읽기 단추를 시험한다. 가짜 녹음기는
  // stop 때 'recording' 글자를 담은 webm 조각 하나를 내놓는다.
  await page.addInitScript(() => {
    Object.defineProperty(window, 'speechSynthesis', { value: { speak() {}, cancel() {} } })
    Object.defineProperty(navigator.mediaDevices, 'getUserMedia', { value: async () => ({ getTracks: () => [{ stop() {} }] }) })
    class Recorder {
      static isTypeSupported() { return true }
      state = 'inactive'
      mimeType = 'audio/webm'
      ondataavailable: ((event: { data: Blob }) => void) | null = null
      onstop: (() => void) | null = null
      start() { this.state = 'recording' }
      stop() {
        this.state = 'inactive'
        this.ondataavailable?.({ data: new Blob(['recording'], { type: this.mimeType }) })
        this.onstop?.()
      }
    }
    Object.defineProperty(window, 'MediaRecorder', { value: Recorder })
  })
  // /api/ 아래 요청에 정해진 가짜 응답을 준다: 사용자 정보(아주 긴 이메일로 좁은 폭 배치를 시험), 가입 201,
  // 로그인·갱신 토큰, 번역 201, 빈 기록, 기록 상세. 목록에 없는 API를 부르면 예외를 던져 검사가 모르는 요청을
  // 놓치지 않게 한다. 경로가 /api/로 시작하는 요청만 잡는다: Vite는 /src/api/ 아래 소스 모듈도 내보내기 때문이다.
  await page.route((url) => url.pathname.startsWith('/api/'), async (route) => {
    const path = new URL(route.request().url()).pathname
    let body: unknown = {}
    let status = 200
    if (path === '/api/auth/me') body = { id: 'user', email: `${'long'.repeat(14)}@example.com` }
    else if (path === '/api/auth/register') status = 201
    else if (path.startsWith('/api/auth/')) body = tokens
    else if (path.startsWith('/api/translate/')) { body = item; status = 201 }
    else if (path === '/api/history') body = { items: [], total: 0 }
    else if (path.startsWith('/api/history/')) body = item
    else throw new Error(`Unexpected API request: ${path}`)
    await route.fulfill({ status, json: body })
  })
})

// 키보드만으로 로그인한다. 로그인 뒤에는 번역 화면 제목으로 포커스가 옮겨져야 한다(화면이 바뀐 것을
// 스크린리더 사용자가 알 수 있게).
async function login(page: Page) {
  await page.goto('/login')
  await expect(page.getByRole('heading', { name: '로그인' })).toBeVisible()
  await tabTo(page, page.getByLabel('이메일'))
  await page.keyboard.type('reader@example.com')
  await page.keyboard.press('Tab')
  await page.keyboard.type('password123')
  await page.keyboard.press('Enter')
  await expect(page.getByRole('heading', { name: '번역', exact: true })).toBeFocused()
}

// 대화 모드: 키보드로 시작·멈춤, 상태 안내(듣는 중, 말소리 감지, 번역 중, 재생 중), 한 마디가 실패해도 대화가
// 이어지는지, 재생 중에는 듣지 않는지(README), 360px 배치를 확인한다.
test('conversation keyboard controls, live states, turn error recovery and 360px layout', async ({ page }) => {
  // 추론·장치 경계만 가짜로 바꾸고, 실제 판정기·세션·화면·API 코드가 돈다. 이 테스트는 ONNX 모델을 올리지
  // 않으며, 실제 브라우저에서 모델을 돌려 보는 확인은 이 테스트의 범위가 아니다.
  // 말소리 판정 모델 실행기 모듈(sileroRuntime.ts)을 가짜 모듈로 바꾸고 불러온 횟수를 센다. 가짜 실행기는
  // 입력의 65번째 표본(앞 문맥 64표본 뒤 현재 조각의 첫 표본) 값을 말소리 확률로 돌려준다. 그래서 진폭 .8로
  // 채운 조각은 말소리, 0인 조각은 조용함으로 판정된다.
  let runtimeLoads = 0
  await page.route((url) => url.pathname === '/src/conversation/sileroRuntime.ts', async (route) => {
    runtimeLoads++
    await route.fulfill({ contentType: 'application/javascript', body: `
      export async function getSileroRunner() {
        return async input => ({ probability: input[64], h: new Float32Array(128), c: new Float32Array(128) });
      }` })
  })
  // 오디오 처리 경계도 가짜로 바꾼다: AudioContext와 AudioWorkletNode 대역을 두고, 검사가 window.capture(말
  // 조각 수, 조용한 조각 수)를 부르면 512표본(32 ms) 조각들을 실제 워클릿처럼 port 메시지로 보낸다. 음성 합성
  // speak는 발화를 잡아 두었다가 finishSpeech로 '읽기 끝'을 알리게 한다(재생이 끝나는 시점을 검사가 정함).
  await page.addInitScript(() => {
    const harness = window as unknown as { capture: (speechFrames: number, quietFrames: number) => void; finishSpeech: () => void }
    class Context {
      sampleRate = 16000; destination = {}; audioWorklet = { addModule: async () => {} }
      async resume() {} async close() {}
      createMediaStreamSource() { return { connect() {}, disconnect() {} } }
    }
    class Node {
      epoch = 0
      port = { onmessage: null as ((event: unknown) => void) | null, close() {},
        postMessage: (data: { epoch: number }) => { this.epoch = data.epoch } }
      connect() {} disconnect() {}
      constructor() {
        harness.capture = (speech, quiet) => {
          const samples = new Float32Array((speech + quiet) * 512)
          samples.fill(.8, 0, speech * 512)
          this.port.onmessage?.({ data: { samples, epoch: this.epoch } })
        }
      }
    }
    Object.defineProperty(window, 'AudioContext', { value: Context })
    Object.defineProperty(window, 'AudioWorkletNode', { value: Node })
    window.speechSynthesis.speak = (utterance) => {
      harness.finishSpeech = () => utterance.onend?.(new SpeechSynthesisEvent('end', { utterance }))
    }
  })
  // 로그인한 실제 모드의 대화 모드는 마디를 웹소켓(WS /api/translate/live, mode 'conversation')으로 흘려보내고
  // end 뒤 서버의 final 또는 error를 기다린다(00bf7d7, docs/api.md). 가짜 웹소켓 서버는 받은 메시지를
  // 기록하고(소리 바이너리는 'audio'로), start의 토큰·방향·mode를 확인한 뒤 ready로 답한다. 첫 마디의 end에는
  // error "No speech was recognized"로, 둘째 마디의 end는 release()를 부를 때까지 붙잡았다가 final로
  // 답한다(번역 중 상태를 검사하려고).
  const received: (string | Record<string, unknown>)[] = []
  let ends = 0
  let release!: () => void
  let closed = false
  await page.routeWebSocket((url) => url.pathname === '/api/translate/live', (ws) => {
    expect(new URL(ws.url()).search).toBe('')
    ws.onClose(() => { closed = true })
    ws.onMessage((raw) => {
      if (typeof raw !== 'string') { received.push('audio'); return }
      const message = JSON.parse(raw)
      received.push(message)
      if (message.type === 'start') {
        expect(message).toEqual({ type: 'start', token: tokens.access_token, source_lang: 'ko', target_lang: 'en', mode: 'conversation' })
        ws.send(JSON.stringify({ type: 'ready' }))
      } else if (message.type === 'end' && ++ends === 1) {
        ws.send(JSON.stringify({ type: 'error', id: message.id, detail: 'No speech was recognized' }))
      } else if (message.type === 'end') {
        release = () => ws.send(JSON.stringify({ type: 'final', id: message.id, result: { ...item, mode: 'speech' } }))
      }
    })
  })
  // 예시 모드만 끝난 마디를 HTTP로 올린다. 실제 모드에서 업로드로 돌아가지 않는지 보려고 업로드 수를 센다.
  let uploads = 0
  await page.route('**/api/translate/speech', async (route) => {
    uploads++
    await route.fulfill({ status: 201, json: { ...item, mode: 'speech' } })
  })
  // 웹소켓으로 보낸 JSON 메시지의 종류와 마디 번호(소리 바이너리는 뺀다).
  const sent = () => received.flatMap((message) => typeof message === 'string' ? [] : [`${message.type}${message.id ?? ''}`])
  const capture = (speech: number, quiet: number) => page.evaluate(([a, b]) => {
    (window as unknown as { capture: (a: number, b: number) => void }).capture(a, b)
  }, [speech, quiet])
  await login(page)
  // 로그인하거나 모드를 고르는 것만으로는 판정 모델 파일을 받지 않는다. 대화 모드를 처음 시작할 때 한 번만
  // 받는다(README: WASM 약 14MB). 모드 단추는 라디오 묶음이라 방향키로 고른다(글자 입력에서 왼쪽 두 번 = 대화
  // 모드).
  expect(runtimeLoads).toBe(0)
  await tabTo(page, page.getByRole('radio', { name: '글자 입력', exact: true }))
  await page.keyboard.press('ArrowLeft')
  await page.keyboard.press('ArrowLeft')
  await expect(page.getByRole('radio', { name: '대화 모드', exact: true })).toBeChecked()
  const panel = page.getByRole('region', { name: '대화 모드' })
  expect(runtimeLoads).toBe(0)
  await audit(page, 'conversation-idle-360')
  // 시작하면 '듣는 중' 상태, 판정 모델을 한 번 불러옴, 대화 중에는 언어 선택이 잠기고 포커스가 '대화 멈춤'
  // 단추로 옮겨진다.
  await activate(page, page.getByRole('button', { name: '대화 시작', exact: true }))
  await expect(panel.getByRole('status')).toContainText('듣는 중')
  expect(runtimeLoads).toBe(1)
  await expect(page.getByLabel('말하거나 입력할 언어')).toBeDisabled()
  await expect(page.getByRole('button', { name: '대화 멈춤' })).toBeFocused()
  // 말 조각 2개면 말 시작으로 판정해 '말소리 감지'를 알린다. 이어서 말 6조각과 조용한 31조각(약 1초)을 보내면
  // 마디가 끝나 업로드되고, 422가 와서 '1번째 말 · 번역 실패'와 안내 문구가 보인다.
  await capture(2, 0)
  await expect(panel.getByRole('status')).toContainText('말소리 감지')
  await capture(6, 31)
  await expect(panel.getByRole('heading', { name: '1번째 말 · 번역 실패' })).toBeVisible()
  await expect(panel.getByText(/말소리를 찾지 못했습니다/)).toBeVisible()
  await audit(page, 'conversation-error-360')
  // 둘째 마디는 결과가 붙잡혀 있어 '듣는 중 · 번역 중'(번역을 기다리는 동안에도 계속 듣는다)이다. 마디마다
  // utterance → (소리) → 192ms 쉼의 pause → 1초 쉼의 end 순서로 보냈다.
  await capture(8, 31)
  await expect(panel.getByRole('status')).toContainText('듣는 중 · 번역 중')
  await expect.poll(() => ends).toBe(2)
  expect(sent()).toEqual(['start', 'utterance1', 'pause1', 'end1', 'utterance2', 'pause2', 'end2'])
  expect(received).toContain('audio')
  await audit(page, 'conversation-translating-360')
  // 결과가 오면 번역 음성을 읽는 동안 '재생 중 · 듣기 멈춤'이 되고, 번역문에는 lang="en"이 붙어 스크린리더가
  // 영어로 읽는다. 재생 중에 들어온 소리는 버리므로 보낸 메시지가 늘지 않는다.
  release()
  await expect(panel.getByRole('status')).toContainText('재생 중 · 듣기 멈춤')
  await expect(panel.getByText('Hello', { exact: true })).toHaveAttribute('lang', 'en')
  const count = received.length
  await capture(8, 31)
  expect(received).toHaveLength(count)
  await audit(page, 'conversation-playing-360')
  // 읽기가 끝나면 다시 '듣는 중'이 되고, '대화 멈춤'을 누르면 웹소켓을 닫고 '멈춤' 상태와 함께 언어 선택이
  // 다시 풀리며 둘째 마디는 '번역 완료'로 남는다. 업로드는 한 번도 없었다.
  await page.evaluate(() => (window as unknown as { finishSpeech: () => void }).finishSpeech())
  await expect(panel.getByRole('status')).toContainText('듣는 중')
  await activate(page, page.getByRole('button', { name: '대화 멈춤' }))
  await expect(panel.getByRole('status')).toHaveText('멈춤')
  await expect.poll(() => closed).toBe(true)
  await expect(page.getByLabel('말하거나 입력할 언어')).toBeEnabled()
  await expect(panel.getByRole('heading', { name: '2번째 말 · 번역 완료' })).toBeVisible()
  expect(uploads).toBe(0)
  await audit(page, 'conversation-stopped-360')
})

// 두 사람 대화: 언어를 자동으로 판별하므로 언어 선택이 없고, 확신이 낮아 방향을 추정한 마디의 안내와 '이 말의
// 방향 바꾸기', 직전 마디 언어 전달, 말풍선 목록과 휴대폰용 마주 보기 화면을 키보드로 확인한다.
test('two-person dialog keyboard, guessed direction, face-to-face view and 360px layout', async ({ page }) => {
  // 대화 모드 검사와 같은 가짜 판정 모델 실행기와 오디오 경계(이름만 captureDialog·finishDialogSpeech로
  // 다름).
  await page.route((url) => url.pathname === '/src/conversation/sileroRuntime.ts', async (route) => {
    await route.fulfill({ contentType: 'application/javascript', body: `
      export async function getSileroRunner() {
        return async input => ({ probability: input[64], h: new Float32Array(128), c: new Float32Array(128) });
      }` })
  })
  await page.addInitScript(() => {
    const harness = window as unknown as { captureDialog: (speechFrames: number, quietFrames: number) => void; finishDialogSpeech: () => void }
    class Context {
      sampleRate = 16000; destination = {}; audioWorklet = { addModule: async () => {} }
      async resume() {} async close() {}
      createMediaStreamSource() { return { connect() {}, disconnect() {} } }
    }
    class Node {
      epoch = 0
      port = { onmessage: null as ((event: unknown) => void) | null, close() {},
        postMessage: (data: { epoch: number }) => { this.epoch = data.epoch } }
      connect() {} disconnect() {}
      constructor() {
        harness.captureDialog = (speech, quiet) => {
          const samples = new Float32Array((speech + quiet) * 512)
          samples.fill(.8, 0, speech * 512)
          this.port.onmessage?.({ data: { samples, epoch: this.epoch } })
        }
      }
    }
    Object.defineProperty(window, 'AudioContext', { value: Context })
    Object.defineProperty(window, 'AudioWorkletNode', { value: Node })
    window.speechSynthesis.speak = (utterance) => {
      harness.finishDialogSpeech = () => utterance.onend?.(new SpeechSynthesisEvent('end', { utterance }))
    }
  })
  // 로그인한 실제 모드의 두 사람 대화는 마디를 웹소켓(mode 'dialog')으로 흘려보내고, 서버가 마디 언어를
  // 판별하며 연결마다 직전 마디 언어를 기억한다(00bf7d7, docs/api.md). 가짜 웹소켓 서버는 받은 메시지를
  // 기록하고, start에 ready로, 마디의 end에 final로 답한다: 첫 마디는 확신 0.55로 한국어라고
  // 추정한(language_guessed) 결과, 둘째 마디는 확신 0.98의 영어 결과.
  const dialogResults = [
    { ...item, id: 'wrong', mode: 'speech', source_lang: 'ko', target_lang: 'en', source_text: '안녕하세요',
      translated_text: 'Hello', language_confidence: .55, language_guessed: true },
    { ...item, id: 'english', mode: 'speech', source_lang: 'en', target_lang: 'ko', source_text: 'Thank you',
      translated_text: '감사합니다', language_confidence: .98, language_guessed: false },
  ]
  const received: (string | Record<string, unknown>)[] = []
  let ends = 0
  let closed = false
  await page.routeWebSocket((url) => url.pathname === '/api/translate/live', (ws) => {
    expect(new URL(ws.url()).search).toBe('')
    ws.onClose(() => { closed = true })
    ws.onMessage((raw) => {
      if (typeof raw !== 'string') { received.push('audio'); return }
      const message = JSON.parse(raw)
      received.push(message)
      if (message.type === 'start') ws.send(JSON.stringify({ type: 'ready' }))
      else if (message.type === 'end') ws.send(JSON.stringify({ type: 'final', id: message.id, result: dialogResults[ends++] }))
    })
  })
  // 예시 모드만 끝난 마디를 HTTP로 올린다. 실제 모드에서 업로드로 돌아가지 않는지 보려고 업로드 수를 센다.
  let uploads = 0
  await page.route('**/api/translate/dialog', async (route) => {
    uploads++
    await route.fulfill({ status: 201, json: dialogResults[0] })
  })
  const messages = () => received.filter((message): message is Record<string, unknown> => typeof message !== 'string')
  // 방향 바꾸기는 같은 녹음을 음성 번역 API에 반대 방향으로 다시 보내고(docs/api.md), 틀린 기록은 DELETE로
  // 지운다. 두 요청을 기록해 확인한다.
  let correctionBody = ''
  await page.route('**/api/translate/speech', async (route) => {
    correctionBody = route.request().postData() ?? ''
    await route.fulfill({ status: 201, json: { ...item, id: 'corrected', mode: 'speech', source_lang: 'en',
      target_lang: 'ko', source_text: 'Hello', translated_text: '안녕하세요' } })
  })
  let deleted = ''
  await page.route('**/api/history/wrong', async (route) => {
    deleted = route.request().method()
    await route.fulfill({ status: 204, body: '' })
  })
  const capture = (speech: number, quiet: number) => page.evaluate(([a, b]) => {
    (window as unknown as { captureDialog: (a: number, b: number) => void }).captureDialog(a, b)
  }, [speech, quiet])
  const finishSpeech = () => page.evaluate(() =>
    (window as unknown as { finishDialogSpeech: () => void }).finishDialogSpeech())

  // 글자 입력에서 오른쪽 두 번 = 두 사람 대화. 말풍선 목록은 aria-live="polite"라 새 말이 붙으면 스크린리더가
  // 읽는다.
  await login(page)
  await tabTo(page, page.getByRole('radio', { name: '글자 입력', exact: true }))
  await page.keyboard.press('ArrowRight'); await page.keyboard.press('ArrowRight')
  await expect(page.getByRole('radio', { name: '두 사람 대화', exact: true })).toBeChecked()
  await expect(page.getByLabel('말하거나 입력할 언어')).toHaveCount(0)
  const panel = page.getByRole('region', { name: '두 사람 대화' })
  await expect(panel.getByLabel('두 사람 대화 내용')).toHaveAttribute('aria-live', 'polite')
  await audit(page, 'dialog-bubbles-idle-360')
  await activate(page, panel.getByRole('button', { name: '두 사람 대화 시작' }))
  await expect(panel.getByRole('status')).toContainText('듣는 중')

  // 첫 마디: 방향을 추정했다는 안내가 보이고, 첫 연결이라 start에 직전 마디 언어(previous_lang·previous_id)가
  // 없다. 이어서 번역 음성을 읽는 동안 듣기를 멈춘다.
  await capture(8, 31)
  await expect(panel.getByText(/방향을 추정했습니다/)).toBeVisible()
  expect(messages()[0]).toEqual({ type: 'start', token: tokens.access_token, mode: 'dialog' })
  await expect(panel.getByRole('status')).toContainText('재생 중 · 듣기 멈춤')
  await finishSpeech()
  await expect(panel.getByRole('status')).toContainText('듣는 중')
  // 둘째 마디: 같은 연결로 보내므로 서버가 첫 마디 언어(ko)를 직전 마디 언어로 써서, 확신이 낮을 때 '번갈아
  // 말한다'고 볼 수 있다(start는 한 번뿐). 말풍선은 언어별 모양(.dialog-turn-ko/-en)으로 나뉜다.
  await capture(8, 31)
  await expect(panel.getByText('감사합니다', { exact: true })).toBeVisible()
  expect(messages().map((message) => `${message.type}${message.id ?? ''}`))
    .toEqual(['start', 'utterance1', 'pause1', 'end1', 'utterance2', 'pause2', 'end2'])
  expect(received).toContain('audio')
  await finishSpeech()
  await expect(panel.locator('.dialog-turn-ko')).toContainText('안녕하세요')
  await expect(panel.locator('.dialog-turn-en')).toContainText('Thank you')
  await audit(page, 'dialog-bubbles-360')

  // 마주 보기 화면: 각 화자 쪽에 자기 언어로 된 번역이 보이고, 맞은편 칸은 내용을 180도 돌려(transform),
  // 휴대폰을 두 사람 사이에 놓았을 때 건너편 사람이 자기 언어 번역을 바로 읽는다(DialogPanel.tsx). '칸 언어
  // 바꾸기'로 두 칸을 바꾸고 '말풍선 보기'로 돌아간다.
  await activate(page, panel.getByRole('button', { name: '마주 보기' }))
  const face = panel.getByLabel('마주 보기 대화')
  await expect(face.getByLabel('한국어 화자 쪽 · 맞은편').locator('.face-translation')).toHaveText('감사합니다')
  await expect(face.getByLabel('영어 화자 쪽 · 가까운 쪽').locator('.face-translation')).toHaveText('Hello')
  await expect(face.getByLabel('한국어 화자 쪽 · 맞은편').locator('.face-pane-content')).not.toHaveCSS('transform', 'none')
  await audit(page, 'dialog-face-to-face-360')
  await activate(page, panel.getByRole('button', { name: '칸 언어 바꾸기' }))
  await expect(face.getByLabel('영어 화자 쪽 · 맞은편')).toBeVisible()
  await activate(page, panel.getByRole('button', { name: '말풍선 보기' }))

  // 방향 바꾸기: 1번째 말이 다시 번역 완료로 바뀌고, 다시 보낸 요청이 en → ko 방향이며, 틀린 기록이 지워지고,
  // 추정 안내가 사라진 뒤 새 번역을 읽는다. 1번째 말은 마지막으로 처리된 마디가 아니라서 서버에 직전 마디
  // 언어를 바꾸라는 previous 메시지를 보내지 않는다. 마지막으로 멈추면 웹소켓을 닫고, 업로드는 한 번도 없었다.
  await activate(page, panel.getByRole('button', { name: '이 말의 방향 바꾸기' }))
  await expect(panel.getByRole('article', { name: '1번째 말 · 번역 완료' })).toContainText('안녕하세요')
  expect(correctionBody).toContain('name="source_lang"')
  expect(correctionBody).toContain('en')
  expect(correctionBody).toContain('name="target_lang"')
  expect(correctionBody).toContain('ko')
  expect(deleted).toBe('DELETE')
  await expect(panel.getByText(/방향을 추정했습니다/)).toHaveCount(0)
  await expect(panel.getByRole('status')).toContainText('재생 중 · 듣기 멈춤')
  expect(messages().some((message) => message.type === 'previous')).toBe(false)
  await finishSpeech()
  await activate(page, panel.getByRole('button', { name: '두 사람 대화 멈춤' }))
  await expect(panel.getByRole('status')).toHaveText('멈춤')
  await expect.poll(() => closed).toBe(true)
  expect(uploads).toBe(0)
  await audit(page, 'dialog-stopped-360')
})

// 동시통역: 가짜 웹소켓 서버로 메시지 흐름(start → utterance → 소리 → pause → end → cancel)을 확인하고,
// 진한·흐린 자막 표시, 마디 오류 뒤 계속하기, 최종 결과 알림, 재생 중 듣기 멈춤, 키보드 시작·멈춤, 360px
// 배치를 본다.
test('live subtitles: keyboard start/stop, stable text, final announcement, playback and narrow layout', async ({ page }) => {
  // 앞의 검사들과 같은 가짜 판정 모델 실행기와 오디오 경계.
  await page.route((url) => url.pathname === '/src/conversation/sileroRuntime.ts', async (route) => {
    await route.fulfill({ contentType: 'application/javascript', body: `
      export async function getSileroRunner() {
        return async input => ({ probability: input[64], h: new Float32Array(128), c: new Float32Array(128) });
      }` })
  })
  await page.addInitScript(() => {
    const harness = window as unknown as { capture: (speech: number, quiet: number) => void; finishSpeech: () => void }
    class Context {
      sampleRate = 16000; destination = {}; audioWorklet = { addModule: async () => {} }
      async resume() {} async close() {}
      createMediaStreamSource() { return { connect() {}, disconnect() {} } }
    }
    class Node {
      epoch = 0
      port = { onmessage: null as ((event: unknown) => void) | null, close() {},
        postMessage: (data: { epoch: number }) => { this.epoch = data.epoch } }
      connect() {} disconnect() {}
      constructor() {
        harness.capture = (speech, quiet) => {
          const samples = new Float32Array((speech + quiet) * 512)
          samples.fill(.8, 0, speech * 512)
          this.port.onmessage?.({ data: { samples, epoch: this.epoch } })
        }
      }
    }
    Object.defineProperty(window, 'AudioContext', { value: Context })
    Object.defineProperty(window, 'AudioWorkletNode', { value: Node })
    window.speechSynthesis.speak = (utterance) => {
      harness.finishSpeech = () => utterance.onend?.(new SpeechSynthesisEvent('end', { utterance }))
    }
  })
  // 가짜 웹소켓 서버: 받은 메시지를 기록하고(소리 바이너리는 'audio'로), reply로 서버 메시지를 보낸다. 토큰이
  // 주소의 쿼리에 붙지 않고(search가 빈 문자열) 첫 start 메시지에 실려 오는지 확인한 뒤 ready로 답한다.
  // 주소의 토큰은 로그에 남기 때문이다(docs/api.md).
  const received: (string | Record<string, unknown>)[] = []
  let reply!: (value: unknown) => void
  let closed = false
  await page.routeWebSocket((url) => url.pathname === '/api/translate/live', (ws) => {
    expect(new URL(ws.url()).search).toBe('')
    reply = (value) => ws.send(JSON.stringify(value))
    ws.onClose(() => { closed = true })
    ws.onMessage((raw) => {
      if (typeof raw !== 'string') { received.push('audio'); return }
      const message = JSON.parse(raw)
      received.push(message)
      if (message.type === 'start') { expect(message.token).toBe(tokens.access_token); reply({ type: 'ready' }) }
    })
  })
  const capture = (speech: number, quiet: number) => page.evaluate(([a, b]) => {
    (window as unknown as { capture: (a: number, b: number) => void }).capture(a, b)
  }, [speech, quiet])
  await login(page)
  // 글자 입력에서 왼쪽 한 번 = 동시통역(마지막 항목으로 돌아감).
  await tabTo(page, page.getByRole('radio', { name: '글자 입력', exact: true }))
  await page.keyboard.press('ArrowLeft')
  await expect(page.getByRole('radio', { name: '동시통역', exact: true })).toBeChecked()
  const panel = page.getByRole('region', { name: '동시통역', exact: true })
  await audit(page, 'live-idle-360')
  // 시작하면 '듣는 중', 포커스는 멈춤 단추로, 언어 선택은 잠긴다.
  await activate(page, page.getByRole('button', { name: '동시통역 시작' }))
  await expect(panel.getByRole('status').first()).toHaveText('듣는 중')
  await expect(page.getByRole('button', { name: '동시통역 멈춤' })).toBeFocused()
  await expect(page.getByLabel('말하거나 입력할 언어')).toBeDisabled()
  // 말 8조각 뒤 조용한 6조각(192 ms)이면 브라우저가 pause를 보낸다. 서버 대역이 원문 자막(앞 5글자 확정)과
  // 번역 자막을 보낸다.
  await capture(8, 6)
  await expect.poll(() => received.at(-1)).toEqual({ type: 'pause', id: 1 })
  reply({ type: 'source', id: 1, text: '안녕하세요 오늘 날씨가', stable: 5 })
  reply({ type: 'translation', id: 1, text: 'Hello, the weather today is', stable: 0 })
  // 자막 목록은 1초마다 바뀌므로 aria-live="off"로 매번 읽지 않는다(LivePanel.tsx). 확정 부분은 <strong>으로
  // 진하게, 나머지는 .live-draft로 흐리게 보이며, 색만으로는 차이를 알 수 없으므로 화면에 보이지 않는 '갱신
  // 중인 부분:' 표시가 원문·번역문에 하나씩 붙는다.
  const captions = panel.getByRole('list', { name: '동시통역 내용' })
  await expect(captions).toHaveAttribute('aria-live', 'off')
  await expect(captions.locator('strong')).toHaveText('안녕하세요')
  await expect(captions.locator('.live-draft').last()).toHaveText('Hello, the weather today is')
  await expect(captions.getByText('갱신 중인 부분:', { exact: true })).toHaveCount(2)
  await audit(page, 'live-subtitles-360')
  // 조용함이 31조각(약 1초)에 이르면 end를 보낸다. 서버가 그 마디의 오류를 보내면 '다음 말은 계속 번역합니다'
  // 경고가 뜨고 연결은 이어진다.
  await capture(0, 25)
  await expect.poll(() => received.at(-1)).toEqual({ type: 'end', id: 1 })
  reply({ type: 'error', id: 1, detail: 'No speech was recognized' })
  await expect(panel.getByRole('alert')).toContainText('다음 말은 계속 번역합니다')
  // 둘째 마디의 최종 결과가 오면 '재생 중 · 듣기 멈춤'이 되고, 스크린리더용 polite 영역이 '2번째 번역 완료.
  // 원문 … 번역문 …'을 한 번 알린다.
  await capture(8, 31)
  await expect.poll(() => received.at(-1)).toEqual({ type: 'end', id: 2 })
  reply({ type: 'final', id: 2, result: { ...item, mode: 'speech', audio_id: null, source_text: '안녕하세요', translated_text: 'Hello' } })
  await expect(panel.getByRole('status').first()).toHaveText('재생 중 · 듣기 멈춤')
  await expect(panel.locator('[aria-live="polite"]')).toContainText('2번째 번역 완료. 원문: 안녕하세요 번역문: Hello')
  // 재생 중에 들어온 소리로는 아무 메시지도 보내지 않는다(받은 메시지 수 그대로).
  const count = received.length
  await capture(8, 31)
  expect(received).toHaveLength(count)
  await audit(page, 'live-final-playing-360')
  // 읽기가 끝나면 다시 듣고, 말 2조각으로 셋째 마디(utterance 3)가 시작된다.
  await page.evaluate(() => (window as unknown as { finishSpeech: () => void }).finishSpeech())
  await expect(panel.getByRole('status').first()).toHaveText('듣는 중')
  await capture(2, 0)
  await expect.poll(() => received.some((message) => typeof message !== 'string' && message.type === 'utterance' && message.id === 3)).toBe(true)
  // 멈추면 진행 중인 셋째 마디를 cancel로 버리고 웹소켓을 닫으며, 언어 선택을 다시 푼다.
  await activate(page, page.getByRole('button', { name: '동시통역 멈춤' }))
  await expect(panel.getByRole('status').first()).toHaveText('멈춤')
  await expect.poll(() => closed).toBe(true)
  expect(received.at(-1)).toEqual({ type: 'cancel', id: 3 })
  await expect(page.getByLabel('말하거나 입력할 언어')).toBeEnabled()
  await audit(page, 'live-stopped-360')
})

// 회원가입·로그인·로그아웃을 키보드로: 첫 Tab에 '본문 바로가기' 링크가 오고 Enter로 본문 제목에 포커스가
// 가는지, 입력 오류와 로그인 실패 때 경고(alert)로 포커스가 옮겨지는지, 화면이 바뀔 때 제목으로 포커스가
// 가는지 확인한다.
test('keyboard signup, skip link, login error, login and logout; labels and contrast', async ({ page }) => {
  await page.goto('/register')
  await expect(page.getByRole('heading', { name: '회원가입' })).toBeVisible()
  await page.keyboard.press('Tab')
  await expect(page.getByRole('link', { name: '본문 바로가기' })).toBeFocused()
  await page.keyboard.press('Enter')
  await expect(page.getByRole('heading', { name: '회원가입' })).toBeFocused()
  await audit(page, 'register-360')
  // 너무 짧은 비밀번호로 가입하면 경고로 포커스가 가고, 비밀번호 입력란의 aria-describedby가 조건 안내와 오류
  // 문단을 함께 가리켜 입력란에서도 이유가 읽힌다.
  await tabTo(page, page.getByLabel('이메일'))
  await page.keyboard.type('reader@example.com')
  await page.keyboard.press('Tab')
  await page.keyboard.type('short')
  await page.keyboard.press('Enter')
  await expect(page.getByRole('alert')).toBeFocused()
  await expect(page.getByLabel('비밀번호')).toHaveAttribute('aria-describedby', 'password-hint auth-error')
  // 비밀번호를 고쳐 가입하면 로그인 화면 제목으로 포커스가 옮겨지고 '가입이 완료' 안내가 보인다.
  await tabTo(page, page.getByLabel('비밀번호'))
  await page.keyboard.press('ControlOrMeta+A')
  await page.keyboard.type('password123')
  await page.keyboard.press('Enter')
  await expect(page.getByRole('heading', { name: '로그인' })).toBeFocused()
  await expect(page.getByRole('status')).toContainText('가입이 완료')
  // 로그인 API가 401을 내면 경고로 포커스가 간다. 가짜 실패를 걷어 낸 뒤 다시 로그인하고, 로그아웃하면 로그인
  // 제목으로 돌아온다.
  await page.route('**/api/auth/login', (route) => route.fulfill({ status: 401, json: { detail: 'Incorrect email or password' } }))
  await tabTo(page, page.getByLabel('이메일'))
  await page.keyboard.type('reader@example.com')
  await page.keyboard.press('Tab')
  await page.keyboard.type('password123')
  await page.keyboard.press('Enter')
  await expect(page.getByRole('alert')).toBeFocused()
  await audit(page, 'login-error-360')
  await page.unroute('**/api/auth/login')
  await tabTo(page, page.getByLabel('비밀번호'))
  await page.keyboard.type('password123')
  await page.keyboard.press('Enter')
  await expect(page.getByRole('heading', { name: '번역', exact: true })).toBeFocused()
  await audit(page, 'translate-empty-360')
  await activate(page, page.getByRole('button', { name: '로그아웃' }))
  await expect(page.getByRole('heading', { name: '로그인' })).toBeFocused()
})

// 글자 번역을 키보드로: 빈 글자로 번역하면 경고와 aria-invalid, 언어 선택을 방향키로 바꾸기, 번역 결과
// 제목으로 포커스 이동, 서버 음성이 없을 때 브라우저 음성으로 읽기·중지(README 설계 판단), 데스크톱 폭에서도
// 검사.
test('keyboard translation, direction, validation, result and speech fallback', async ({ page }) => {
  await login(page)
  await activate(page, page.getByRole('button', { name: '번역하기' }))
  await expect(page.getByRole('alert')).toBeFocused()
  await expect(page.getByLabel(/번역할 글/)).toHaveAttribute('aria-invalid', 'true')
  await audit(page, 'translation-error-360')
  await tabTo(page, page.getByLabel('말하거나 입력할 언어'))
  await page.keyboard.press('ArrowDown')
  await expect(page.getByLabel('말하거나 입력할 언어')).toHaveValue('en')
  await tabTo(page, page.getByLabel(/번역할 글/))
  await page.keyboard.insertText('Hello')
  await activate(page, page.getByRole('button', { name: '번역하기' }))
  await expect(page.getByRole('heading', { name: '번역 결과' })).toBeFocused()
  await activate(page, page.getByRole('button', { name: '번역문 읽기' }))
  await activate(page, page.getByRole('button', { name: '읽기 중지' }))
  await audit(page, 'translation-result-360')
  // 같은 결과 화면을 데스크톱 폭(1280px)에서도 검사한다(README의 데스크톱 화면 그림).
  await page.setViewportSize({ width: 1280, height: 900 })
  await audit(page, 'translation-result-desktop')
})

// 음성 입력을 키보드로: 녹음 시작 → 포커스가 '녹음 취소'로, 취소 → 다시 '녹음 시작'으로, 녹음 끝내기 → 준비
// 안내, 번역. 이어서 음성 파일 고르기 단추(운영체제 파일 창만 대역)와 아주 긴 파일 이름 표시, 방향을 바꾸면
// 고른 음성이 비워지는지, 같은 파일을 다시 골라도 선택되는지 확인한다.
test('keyboard recording start, cancel, stop, upload chooser and long file name', async ({ page }) => {
  await login(page)
  await tabTo(page, page.getByRole('radio', { name: '글자 입력', exact: true }))
  await page.keyboard.press('ArrowRight')
  await expect(page.getByRole('radio', { name: '음성 입력', exact: true })).toBeChecked()
  await activate(page, page.getByRole('button', { name: '녹음 시작' }))
  await expect(page.getByRole('button', { name: '녹음 취소' })).toBeFocused()
  await page.keyboard.press('Enter')
  await expect(page.getByRole('button', { name: '녹음 시작' })).toBeFocused()
  await page.keyboard.press('Enter')
  await activate(page, page.getByRole('button', { name: '녹음 끝내기' }))
  await expect(page.getByRole('button', { name: '녹음 시작' })).toBeFocused()
  await expect(page.getByText(/번역할 준비가 되었습니다/)).toBeVisible()
  await activate(page, page.getByRole('button', { name: '번역하기' }))
  await expect(page.getByRole('heading', { name: '번역 결과' })).toBeFocused()
  // 파일 단추의 접근 가능한 설명에 지금 고른 음성(녹음 완료 또는 파일 이름)이 들어가, 스크린리더 사용자도
  // 무엇이 선택됐는지 알 수 있다.
  const fileButton = page.getByRole('button', { name: '또는 음성 파일 선택' })
  await expect(fileButton).toHaveAccessibleDescription(/녹음 완료/)
  await tabTo(page, fileButton)
  const chooser = page.waitForEvent('filechooser')
  await page.keyboard.press('Enter')
  // 운영체제의 파일 고르기 창만 가짜로 바꾸고, 단추는 Enter로 누른다.
  const file = { name: `${'long-filename-'.repeat(20)}.wav`, mimeType: 'audio/wav', buffer: Buffer.from('recording') }
  await (await chooser).setFiles(file)
  // 실제 파일 입력은 숨기고 단추로 연다. 선택 표시 영역은 aria-atomic이라 바뀔 때 전체가 한 번에 읽힌다.
  await expect(page.getByLabel('음성 파일', { exact: true })).toBeHidden()
  await expect(fileButton).toHaveAccessibleDescription(new RegExp(`선택한 파일: ${file.name}`))
  await expect(page.locator('#audio-selection')).toContainText(file.name)
  await expect(page.locator('#audio-selection')).toHaveAttribute('aria-atomic', 'true')
  await audit(page, 'audio-file-360')
  // 방향을 바꿔 고른 음성을 비운 뒤, 똑같은 파일을 다시 고른다.
  await tabTo(page, page.getByLabel('말하거나 입력할 언어'))
  await page.keyboard.press('ArrowDown')
  await expect(page.locator('#audio-selection')).toHaveText('선택한 음성이 없습니다.')
  await expect(page.getByRole('button', { name: '번역하기' })).toBeDisabled()
  await tabTo(page, fileButton)
  const sameFileChooser = page.waitForEvent('filechooser')
  await page.keyboard.press('Space')
  await (await sameFileChooser).setFiles(file)
  await expect(page.locator('#audio-selection')).toContainText(file.name)
  await activate(page, page.getByRole('button', { name: '번역하기' }))
  await expect(page.getByRole('heading', { name: '번역 결과' })).toBeFocused()
})

// 기록 화면을 키보드로: 페이지 넘기기(전체 21개, 한 페이지 20개), 상세 열기, 브라우저 기본 audio 컨트롤을
// Space로 재생·정지, 삭제 확인 창에서 취소·확인, 빈 상태와 첫 번역 링크.
test('keyboard history pages, detail, native audio, delete cancel/confirm and empty state', async ({ page }) => {
  await login(page)
  // 기록 목록은 offset=20이면 둘째 페이지의 'last' 항목을, 삭제 뒤에는 빈 목록을 준다. 'last' 상세는 번역
  // 음성이 있는 기록이다.
  let removed = false
  await page.route('**/api/history?**', (route) => {
    const second = new URL(route.request().url()).searchParams.get('offset') === '20'
    return route.fulfill({ json: removed ? { items: [], total: 0 } : { items: [{ ...item, id: second ? 'last' : 'entry' }], total: 21 } })
  })
  await page.route('**/api/history/last', (route) => {
    if (route.request().method() === 'DELETE') { removed = true; return route.fulfill({ status: 204 }) }
    return route.fulfill({ json: { ...item, id: 'last', audio_id: 'audio' } })
  })
  // 10초짜리 무음 WAV(16 kHz 16비트 모노)를 44바이트 머리부터 직접 써서, 브라우저 audio 요소가 실제로 열고
  // 재생할 수 있는 음성을 준다.
  const wav = Buffer.alloc(44 + 16000 * 2 * 10)
  wav.write('RIFF'); wav.writeUInt32LE(wav.length - 8, 4); wav.write('WAVEfmt ', 8)
  wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20); wav.writeUInt16LE(1, 22)
  wav.writeUInt32LE(16000, 24); wav.writeUInt32LE(32000, 28); wav.writeUInt16LE(2, 32); wav.writeUInt16LE(16, 34)
  wav.write('data', 36); wav.writeUInt32LE(wav.length - 44, 40)
  await page.route('**/api/audio/audio', (route) => route.fulfill({ contentType: 'audio/wav', body: wav }))
  await activate(page, page.getByRole('link', { name: '기록', exact: true }))
  await expect(page.getByRole('heading', { name: '번역 기록' })).toBeFocused()
  await activate(page, page.getByRole('button', { name: '다음 페이지' }))
  await expect(page.getByText(/21–21번째/)).toBeVisible()
  await activate(page, page.getByRole('button', { name: '이전 페이지' }))
  await expect(page.getByText(/1–1번째/)).toBeVisible()
  await activate(page, page.getByRole('button', { name: '다음 페이지' }))
  // 상세 화면 제목으로 포커스가 가고, audio 컨트롤에 Tab으로 가서 데이터가 준비되면(readyState > 1) Space로
  // 재생과 정지가 되는지 본다.
  await activate(page, page.locator('.history-link'))
  await expect(page.getByRole('heading', { name: '기록 상세' })).toBeFocused()
  await tabTo(page, page.getByLabel('번역 음성'))
  await expect.poll(() => page.locator('audio').evaluate((el) => el.readyState)).toBeGreaterThan(1)
  await page.keyboard.press('Space')
  await expect.poll(() => page.locator('audio').evaluate((el) => el.paused)).toBe(false)
  await page.keyboard.press('Space')
  await expect.poll(() => page.locator('audio').evaluate((el) => el.paused)).toBe(true)
  await audit(page, 'history-detail-360')
  // 삭제 확인 창에서 취소하면 지우지 않고, 확인하면 지운 뒤 빈 기록 목록으로 돌아가 제목에 포커스가 간다. 빈
  // 상태의 '첫 번역 시작하기' 링크로 번역 화면에 간다.
  page.once('dialog', (dialog) => dialog.dismiss())
  await activate(page, page.getByRole('button', { name: '기록 삭제' }))
  expect(removed).toBe(false)
  page.once('dialog', (dialog) => dialog.accept())
  await page.keyboard.press('Enter')
  await expect(page.getByText('아직 번역 기록이 없습니다.', { exact: false })).toBeVisible()
  await expect(page.getByRole('heading', { name: '번역 기록' })).toBeFocused()
  await audit(page, 'history-empty-360')
  await activate(page, page.getByRole('link', { name: '첫 번역 시작하기' }))
  await expect(page.getByRole('heading', { name: '번역', exact: true })).toBeFocused()
})

// 오류 상태: 기록 목록이 503이면 경고와 '기록 다시 불러오기' 단추가 보이고 다시 시도하면 빈 목록이 뜬다. 번역
// 음성이 404면 경고와 '음성 다시 불러오기' 단추가 보인다. 오류 화면도 360px에서 검사한다.
test('history and audio errors expose retry controls, narrow layout remains readable', async ({ page }) => {
  await login(page)
  await page.route('**/api/history?**', (route) => route.fulfill({ status: 503, json: {} }))
  await activate(page, page.getByRole('link', { name: '기록', exact: true }))
  await expect(page.getByRole('alert')).toBeVisible()
  await audit(page, 'history-error-360')
  await page.unroute('**/api/history?**')
  await activate(page, page.getByRole('button', { name: '기록 다시 불러오기' }))
  await expect(page.getByText(/아직 번역 기록이 없습니다/)).toBeVisible()
  await page.route('**/api/history/entry', (route) => route.fulfill({ json: { ...item, audio_id: 'missing' } }))
  await page.route('**/api/audio/missing', (route) => route.fulfill({ status: 404, json: { detail: 'Audio not found' } }))
  await page.goto('/history/entry')
  await expect(page.getByRole('alert')).toBeVisible()
  await activate(page, page.getByRole('button', { name: '음성 다시 불러오기' }))
  await expect(page.getByRole('alert')).toBeVisible()
  await audit(page, 'audio-error-360')
})

import { test, expect, type Page, type Locator } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'

const tokens = { access_token: 'access', refresh_token: 'refresh', expires_in: 1800 }
const item = {
  id: 'entry', mode: 'text', source_lang: 'ko', target_lang: 'en', source_text: '안녕하세요',
  translated_text: 'Hello', audio_id: null, tts_error: 'Speech synthesis is not available',
  stt_ms: null, mt_ms: 120, tts_ms: null, created_at: '2026-09-13T00:00:00Z',
}

// Reach every control using the browser's Tab order, never locator.focus/click.
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

async function activate(page: Page, target: Locator) {
  await tabTo(page, target)
  await page.keyboard.press('Enter')
}

async function audit(page: Page, name: string) {
  const results = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa', 'wcag22aa']).analyze()
  expect(results.violations, `${name}: accessibility`).toEqual([])
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), `${name}: horizontal overflow`).toBe(true)
  await page.screenshot({ path: test.info().outputPath(`${name}.png`), fullPage: true })
}

test.beforeEach(async ({ page }) => {
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
  // Match API paths only; Vite also serves source modules under /src/api/.
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

test('keyboard signup, skip link, login error, login and logout; labels and contrast', async ({ page }) => {
  await page.goto('/register')
  await expect(page.getByRole('heading', { name: '회원가입' })).toBeVisible()
  await page.keyboard.press('Tab')
  await expect(page.getByRole('link', { name: '본문 바로가기' })).toBeFocused()
  await page.keyboard.press('Enter')
  await expect(page.getByRole('heading', { name: '회원가입' })).toBeFocused()
  await audit(page, 'register-360')
  await tabTo(page, page.getByLabel('이메일'))
  await page.keyboard.type('reader@example.com')
  await page.keyboard.press('Tab')
  await page.keyboard.type('short')
  await page.keyboard.press('Enter')
  await expect(page.getByRole('alert')).toBeFocused()
  await expect(page.getByLabel('비밀번호')).toHaveAttribute('aria-describedby', 'password-hint auth-error')
  await tabTo(page, page.getByLabel('비밀번호'))
  await page.keyboard.press('ControlOrMeta+A')
  await page.keyboard.type('password123')
  await page.keyboard.press('Enter')
  await expect(page.getByRole('heading', { name: '로그인' })).toBeFocused()
  await expect(page.getByRole('status')).toContainText('가입이 완료')
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
  await page.setViewportSize({ width: 1280, height: 900 })
  await audit(page, 'translation-result-desktop')
})

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
  const fileButton = page.getByRole('button', { name: '또는 음성 파일 선택' })
  await expect(fileButton).toHaveAccessibleDescription(/녹음 완료/)
  await tabTo(page, fileButton)
  const chooser = page.waitForEvent('filechooser')
  await page.keyboard.press('Enter')
  // Only the operating system file picker is replaced; the button is activated with Enter.
  const file = { name: `${'long-filename-'.repeat(20)}.wav`, mimeType: 'audio/wav', buffer: Buffer.from('recording') }
  await (await chooser).setFiles(file)
  await expect(page.getByLabel('음성 파일', { exact: true })).toBeHidden()
  await expect(fileButton).toHaveAccessibleDescription(new RegExp(`선택한 파일: ${file.name}`))
  await expect(page.locator('#audio-selection')).toContainText(file.name)
  await expect(page.locator('#audio-selection')).toHaveAttribute('aria-atomic', 'true')
  await audit(page, 'audio-file-360')
  // Change direction to clear the clip, then pick the exact same file again.
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

test('keyboard history pages, detail, native audio, delete cancel/confirm and empty state', async ({ page }) => {
  await login(page)
  let removed = false
  await page.route('**/api/history?**', (route) => {
    const second = new URL(route.request().url()).searchParams.get('offset') === '20'
    return route.fulfill({ json: removed ? { items: [], total: 0 } : { items: [{ ...item, id: second ? 'last' : 'entry' }], total: 21 } })
  })
  await page.route('**/api/history/last', (route) => {
    if (route.request().method() === 'DELETE') { removed = true; return route.fulfill({ status: 204 }) }
    return route.fulfill({ json: { ...item, id: 'last', audio_id: 'audio' } })
  })
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
  await activate(page, page.locator('.history-link'))
  await expect(page.getByRole('heading', { name: '기록 상세' })).toBeFocused()
  await tabTo(page, page.getByLabel('번역 음성'))
  await expect.poll(() => page.locator('audio').evaluate((el) => el.readyState)).toBeGreaterThan(1)
  await page.keyboard.press('Space')
  await expect.poll(() => page.locator('audio').evaluate((el) => el.paused)).toBe(false)
  await page.keyboard.press('Space')
  await expect.poll(() => page.locator('audio').evaluate((el) => el.paused)).toBe(true)
  await audit(page, 'history-detail-360')
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

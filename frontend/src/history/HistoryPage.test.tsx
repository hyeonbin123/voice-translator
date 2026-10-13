// [파일 역할] 번역 기록 화면(HistoryPage) 테스트 묶음. 앱 전체 라우트(AppRoutes)를 StrictMode와
// MemoryRouter로 그려, 로그인 복원 → 목록·페이지 이동 → 상세·음성 → 삭제까지 실제 사용 흐름대로 확인한다.
// [무엇을 지키나] 인증 머리글이 붙은 요청, 페이지 주소 유지, 음성 blob URL 해제, 삭제 확인과 중복 방지, 서버
// 응답 내용을 화면에 그대로 내지 않기, 화면을 떠날 때 요청 취소, 잘못된 offset 바로잡기.
// [가짜 객체] fetch를 경로별로 응답하는 가짜로 바꾸고(목록 21개 = 20 + 1, 두 페이지),
// speechSynthesis·URL.createObjectURL·confirm·미디어 pause처럼 jsdom에 없거나 화면을 막는 브라우저 기능을
// 가짜로 둔다.

import { StrictMode } from 'react'
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { AppRoutes } from '../App'
import { ApiClient, REFRESH_KEY } from '../api/client'
import type { HistoryItem } from './api'

// 테스트 데이터: first는 서버 음성이 있는 음성 번역, last는 음성 없는 기록(두 번째 페이지의 유일한 항목),
// firstPage는 첫 페이지 20개.
const first: HistoryItem = {
  id: '080b161c-53d6-4460-992b-f778a5e348cd', mode: 'speech', source_lang: 'ko', target_lang: 'en',
  source_text: '안녕하세요', translated_text: 'Hello', stt_model: 'stt', mt_model: 'mt', tts_model: 'tts',
  stt_ms: 200, mt_ms: 120, tts_ms: 300, audio_id: 'audio-1', created_at: '2026-09-13T00:00:00Z',
}
const last: HistoryItem = { ...first, id: 'last', source_text: '감사합니다', translated_text: 'Thank you', audio_id: null }
const firstPage = [first, ...Array.from({ length: 19 }, (_, i) => ({
  ...first, id: `item-${i}`, source_text: `이전 원문 ${i}`, translated_text: `Earlier translation ${i}`, audio_id: null,
}))]
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status })
const fetchMock = vi.fn<typeof fetch>()
const speak = vi.fn()
const cancel = vi.fn()
class Utterance {
  text: string
  lang = ''
  constructor(text: string) { this.text = text }
}
let client: ApiClient

// refresh 토큰을 미리 넣어 화면이 열리면 세션 복원부터 하게 하고, 알 수 없는 요청은 오류로 던져 예상 밖
// 호출을 바로 드러낸다.
beforeEach(() => {
  sessionStorage.clear()
  sessionStorage.setItem(REFRESH_KEY, 'refresh')
  client = new ApiClient()
  speak.mockReset(); cancel.mockReset(); fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
  vi.stubGlobal('speechSynthesis', { speak, cancel })
  vi.stubGlobal('SpeechSynthesisUtterance', Utterance)
  vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => undefined)
  Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: vi.fn(() => 'blob:history') })
  Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: vi.fn() })
  vi.spyOn(window, 'confirm').mockReturnValue(true)
  fetchMock.mockImplementation(async (path, init) => {
    if (path === '/api/auth/refresh') return json({ access_token: 'access', refresh_token: 'refresh' })
    if (path === '/api/auth/me') return json({ id: 'user', email: 'person@example.com' })
    if (path === '/api/history?limit=20&offset=0') return json({ items: firstPage, total: 21 })
    if (path === '/api/history?limit=20&offset=20') return json({ items: [last], total: 21 })
    if (init?.method === 'DELETE') return new Response(null, { status: 204 })
    if (path === `/api/history/${first.id}`) return json(first)
    if (path === '/api/history/last') return json(last)
    if (path === '/api/audio/audio-1') return new Response('wav', { headers: { 'Content-Type': 'audio/wav' } })
    throw new Error(`Unexpected request: ${path}`)
  })
})
afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals() })

// show: 주어진 주소로 앱을 그린다. listCalls·deletes: 목록 요청과 삭제 요청만 골라 보기.
function show(path = '/history') {
  return render(<StrictMode><MemoryRouter initialEntries={[path]}><AppRoutes client={client} /></MemoryRouter></StrictMode>)
}
const listCalls = () => fetchMock.mock.calls.filter(([path]) => String(path).startsWith('/api/history?'))
const deletes = () => fetchMock.mock.calls.filter(([, init]) => init?.method === 'DELETE')

// 확인: 첫 페이지(1–20번째)와 다음 페이지(21–21번째)의 범위 표시와 버튼 잠김, 상세로 갔다가 '기록 목록으로'가
// 보던 둘째 페이지로 돌아오는지, 서버 음성이 없는 기록은 브라우저 음성(en-US)으로 읽고 화면을 떠나면 읽기를
// 멈추는지, 목록 요청마다 Bearer 토큰이 붙는지.
it('fetches authenticated pages with total, follows detail, and returns to the same page', async () => {
  show()
  await screen.findByText('전체 21개 · 1–20번째')
  expect(screen.getByRole('button', { name: '이전 페이지' })).toBeDisabled()
  await userEvent.click(screen.getByRole('button', { name: '다음 페이지' }))
  await screen.findByText('전체 21개 · 21–21번째')
  expect(screen.getByRole('button', { name: '다음 페이지' })).toBeDisabled()
  await userEvent.click(screen.getByRole('link', { name: /감사합니다/ }))
  await screen.findByRole('heading', { name: '번역문' })
  expect(screen.getByText('Thank you')).toHaveAttribute('lang', 'en')
  await userEvent.click(screen.getByRole('button', { name: '번역문 읽기' }))
  expect(speak.mock.calls[0][0]).toMatchObject({ text: 'Thank you', lang: 'en-US' })
  await userEvent.click(screen.getByRole('link', { name: '기록 목록으로' }))
  await screen.findByText('전체 21개 · 21–21번째')
  expect(cancel).toHaveBeenCalled()
  for (const [, init] of listCalls()) expect(new Headers(init?.headers).get('Authorization')).toBe('Bearer access')
  await userEvent.click(screen.getByRole('button', { name: '이전 페이지' }))
  await screen.findByText('전체 21개 · 1–20번째')
})

// 확인: 상세 주소로 바로 들어와도 열리고, 번역 음성은 인증 머리글을 붙여 받아 blob URL로 재생하며, 화면을
// 떠나면 그 URL을 해제한다. 단계별 시간(200 ms, 300 ms)도 보인다.
it('loads detail directly, fetches audio with auth, and revokes its blob URL on leaving', async () => {
  const view = show(`/history/${first.id}`)
  await waitFor(() => expect(screen.getByLabelText('번역 음성')).toHaveAttribute('src', 'blob:history'))
  expect(screen.getByText('200 ms')).toBeInTheDocument()
  expect(screen.getByText('300 ms')).toBeInTheDocument()
  const [, init] = fetchMock.mock.calls.find(([path]) => path === '/api/audio/audio-1')!
  expect(new Headers(init?.headers).get('Authorization')).toBe('Bearer access')
  expect(URL.createObjectURL).toHaveBeenCalledWith(expect.any(Blob))
  view.unmount()
  expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:history')
})

// 확인: 확인 창에서 취소하면 삭제 요청을 보내지 않는다. 확인하면 DELETE를 보내고, 그 결과 마지막
// 페이지(offset 20)가 비면 마지막으로 남은 첫 페이지로 옮겨 간다.
it('requires confirmation and handles 204 deletion by reloading and correcting an empty last page', async () => {
  const original = fetchMock.getMockImplementation()!
  let removed = false
  fetchMock.mockImplementation(async (path, init) => {
    if (init?.method === 'DELETE') { removed = true; return new Response(null, { status: 204 }) }
    if (removed && path === '/api/history?limit=20&offset=20') return json({ items: [], total: 20 })
    if (removed && path === '/api/history?limit=20&offset=0') return json({ items: firstPage, total: 20 })
    return original(path, init)
  })
  show('/history/last?offset=20')
  await screen.findByText('Thank you')
  vi.mocked(window.confirm).mockReturnValueOnce(false)
  await userEvent.click(screen.getByRole('button', { name: '기록 삭제' }))
  expect(deletes()).toHaveLength(0)
  expect(window.confirm).toHaveBeenCalledWith('이 번역 기록과 연결된 음성을 삭제할까요?')
  await userEvent.click(screen.getByRole('button', { name: '기록 삭제' }))
  await screen.findByText('전체 20개 · 1–20번째')
  expect(deletes()).toHaveLength(1)
  expect(deletes()[0][0]).toBe('/api/history/last')
  expect(new Headers(deletes()[0][1]?.headers).get('Authorization')).toBe('Bearer access')
})

// 확인: 삭제가 500으로 실패하면 상세를 그대로 두고 일반 오류 문구를 보이며 다시 누를 수 있다. 응답의 detail은
// 화면에 나오지 않는다.
it('keeps the detail and allows retry after deletion fails without exposing response input', async () => {
  show('/history/last')
  await screen.findByText('Thank you')
  fetchMock.mockResolvedValueOnce(json({ detail: 'private-input' }, 500))
  await userEvent.click(screen.getByRole('button', { name: '기록 삭제' }))
  expect(await screen.findByRole('alert')).toHaveTextContent('요청을 처리하지 못했습니다')
  expect(screen.queryByText('private-input')).not.toBeInTheDocument()
  expect(screen.getByRole('button', { name: '기록 삭제' })).toBeEnabled()
  expect(screen.getByText('Thank you')).toBeInTheDocument()
})

// 확인: 삭제 버튼을 두 번 눌러도 요청은 하나만 가고, 그동안 버튼은 '삭제 중…'으로 잠긴다. 끝나면 목록으로
// 간다.
it('prevents duplicate deletion while a request is pending', async () => {
  show('/history/last')
  await screen.findByText('Thank you')
  let resolve!: (response: Response) => void
  fetchMock.mockReturnValueOnce(new Promise((done) => { resolve = done }))
  await userEvent.dblClick(screen.getByRole('button', { name: '기록 삭제' }))
  expect(screen.getByRole('button', { name: '삭제 중…' })).toBeDisabled()
  expect(deletes()).toHaveLength(1)
  await act(async () => resolve(new Response(null, { status: 204 })))
  await screen.findByRole('heading', { name: '번역 기록' })
})

// 확인: 목록 요청이 실패하면 오류와 '기록 다시 불러오기'를 보이고, 다시 불러와 기록이 없으면 첫 번역 링크를
// 보인다.
it('shows empty history and retries a failed list request', async () => {
  const original = fetchMock.getMockImplementation()!
  let failed = true
  fetchMock.mockImplementation(async (path, init) => {
    if (String(path).startsWith('/api/history?')) return failed ? json({ detail: 'private' }, 500) : json({ items: [], total: 0 })
    return original(path, init)
  })
  show()
  await screen.findByRole('alert')
  failed = false
  await userEvent.click(screen.getByRole('button', { name: '기록 다시 불러오기' }))
  await screen.findByText(/아직 번역 기록이 없습니다/)
  expect(screen.getByRole('button', { name: '다음 페이지' })).toBeDisabled()
  expect(screen.getByRole('link', { name: '첫 번역 시작하기' })).toHaveAttribute('href', '/translate')
})

// 확인: 없는(또는 남의) 기록은 404로 '기록이 없거나 삭제되었습니다'를 보이고, 응답 내용은 숨기며, 목록으로
// 돌아갈 링크를 남긴다.
it('shows a missing detail safely and offers a way back', async () => {
  const original = fetchMock.getMockImplementation()!
  fetchMock.mockImplementation(async (path, init) => path === '/api/history/missing'
    ? json({ detail: 'private-input' }, 404) : original(path, init))
  show('/history/missing')
  expect(await screen.findByRole('alert')).toHaveTextContent('기록이 없거나 삭제되었습니다')
  expect(screen.queryByText('private-input')).not.toBeInTheDocument()
  expect(screen.getByRole('link', { name: '기록 목록으로' })).toBeInTheDocument()
})

// 확인: 번역 음성만 404('Audio not found')면 계약 문구('번역 음성이 없거나 삭제되었습니다')를 보이고, 기록
// 내용은 그대로 보인다.
it('shows the audio 404 contract error on a history detail', async () => {
  const original = fetchMock.getMockImplementation()!
  fetchMock.mockImplementation(async (path, init) => path === '/api/audio/audio-1'
    ? json({ detail: 'Audio not found', input: 'private-input' }, 404) : original(path, init))
  show(`/history/${first.id}`)
  expect(await screen.findByRole('alert')).toHaveTextContent('번역 음성이 없거나 삭제되었습니다')
  expect(screen.queryByText('private-input')).not.toBeInTheDocument()
  expect(screen.getByText('Hello')).toBeInTheDocument()
})

// 확인: 목록이 401이고 토큰 갱신도 401로 거절되면 refresh 토큰을 지우고 로그인 화면으로 간다. 보호된 내용은
// 보이지 않는다.
it('redirects to login when history auth cannot refresh', async () => {
  const original = fetchMock.getMockImplementation()!
  let refreshes = 0
  fetchMock.mockImplementation(async (path, init) => {
    if (String(path).startsWith('/api/history?')) return json({}, 401)
    if (path === '/api/auth/refresh' && refreshes++ > 0) return json({}, 401)
    return original(path, init)
  })
  show()
  await screen.findByRole('heading', { name: '로그인' })
  expect(sessionStorage.getItem(REFRESH_KEY)).toBeNull()
  expect(screen.queryByText('Hello')).not.toBeInTheDocument()
})

// 확인: 목록을 받는 중에 번역 화면으로 이동하면 요청을 취소하고, 늦게 온 응답은 화면에 반영하지 않는다.
it('aborts a pending list when leaving and ignores a late response', async () => {
  const original = fetchMock.getMockImplementation()!
  let resolve!: (response: Response) => void
  const pending = new Promise<Response>((done) => { resolve = done })
  fetchMock.mockImplementation(async (path, init) => String(path).startsWith('/api/history?') ? pending : original(path, init))
  show()
  await screen.findByText('기록을 불러오는 중…')
  await waitFor(() => expect(listCalls().length).toBeGreaterThan(0))
  await userEvent.click(screen.getByRole('link', { name: '번역' }))
  expect(listCalls().every(([, init]) => init?.signal?.aborted)).toBe(true)
  await act(async () => resolve(json({ items: [first], total: 1 })))
  expect(screen.getByRole('heading', { name: '번역' })).toBeInTheDocument()
  expect(screen.queryByText('Hello')).not.toBeInTheDocument()
})

// 확인: 음수, 숫자가 아닌 값, 안전한 정수 범위를 넘는 값은 offset 0으로 바로잡아 서버에 보낸다(서버는 잘못된
// 값에 422를 준다).
it.each(['-20', 'nope', '9007199254740992'])('normalizes invalid offset %s before requesting the API', async (offset) => {
  show(`/history?offset=${offset}`)
  await screen.findByText('Hello')
  expect(listCalls().every(([path]) => path === '/api/history?limit=20&offset=0')).toBe(true)
})

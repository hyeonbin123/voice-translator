import { StrictMode } from 'react'
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { AppRoutes } from '../App'
import { ApiClient, REFRESH_KEY } from '../api/client'
import type { HistoryItem } from './api'

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

function show(path = '/history') {
  return render(<StrictMode><MemoryRouter initialEntries={[path]}><AppRoutes client={client} /></MemoryRouter></StrictMode>)
}
const listCalls = () => fetchMock.mock.calls.filter(([path]) => String(path).startsWith('/api/history?'))
const deletes = () => fetchMock.mock.calls.filter(([, init]) => init?.method === 'DELETE')

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

it('shows a missing detail safely and offers a way back', async () => {
  const original = fetchMock.getMockImplementation()!
  fetchMock.mockImplementation(async (path, init) => path === '/api/history/missing'
    ? json({ detail: 'private-input' }, 404) : original(path, init))
  show('/history/missing')
  expect(await screen.findByRole('alert')).toHaveTextContent('기록이 없거나 삭제되었습니다')
  expect(screen.queryByText('private-input')).not.toBeInTheDocument()
  expect(screen.getByRole('link', { name: '기록 목록으로' })).toBeInTheDocument()
})

it('shows the audio 404 contract error on a history detail', async () => {
  const original = fetchMock.getMockImplementation()!
  fetchMock.mockImplementation(async (path, init) => path === '/api/audio/audio-1'
    ? json({ detail: 'Audio not found', input: 'private-input' }, 404) : original(path, init))
  show(`/history/${first.id}`)
  expect(await screen.findByRole('alert')).toHaveTextContent('번역 음성이 없거나 삭제되었습니다')
  expect(screen.queryByText('private-input')).not.toBeInTheDocument()
  expect(screen.getByText('Hello')).toBeInTheDocument()
})

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

it.each(['-20', 'nope', '9007199254740992'])('normalizes invalid offset %s before requesting the API', async (offset) => {
  show(`/history?offset=${offset}`)
  await screen.findByText('Hello')
  expect(listCalls().every(([path]) => path === '/api/history?limit=20&offset=0')).toBe(true)
})

// [파일 역할] 앱 뼈대(App.tsx의 AppRoutes)와 로그인·가입 화면 테스트 묶음. 실제 ApiClient에 가짜 fetch를
// 붙이고 MemoryRouter로 주소를 정해, 보호 화면 막기, 로그인 뒤 원래 화면으로 돌아가기, 세션 복원, 예시 모드
// 전환, 가입 검증, 오류 문구를 확인한다.
// [왜 StrictMode인가] 실제 앱(main.tsx)처럼 StrictMode로 감싸 effect가 두 번 돌아도 세션 복원 요청이 한 번만
// 나가는지까지 본다.
// [가짜 객체] fetch는 테스트마다 응답 순서를 정하고, VITE_TRANSLATION_MOCK 환경 변수는 vi.stubEnv로 바꿔 예시
// 모드와 실제 모드를 오간다.

import { StrictMode } from 'react'
import { act, fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { AppRoutes } from './App'
import { ApiClient, REFRESH_KEY } from './api/client'

const person = { id: 'user-1', email: 'person@example.com', created_at: '2026-09-13T00:00:00Z' }
const tokens = { access_token: 'access', refresh_token: 'refresh', token_type: 'bearer', expires_in: 1800 }
const response = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status })
const fetchMock = vi.fn<typeof fetch>()
let client: ApiClient

beforeEach(() => {
  vi.stubEnv('VITE_TRANSLATION_MOCK', undefined)
  sessionStorage.clear()
  localStorage.clear()
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
  client = new ApiClient()
})
afterEach(() => { vi.unstubAllGlobals(); vi.unstubAllEnvs() })

// show: 주어진 주소로 앱을 그린다. fill: 이메일과 비밀번호를 사용자가 치듯 입력한다.
function show(path = '/login') {
  return render(<StrictMode><MemoryRouter initialEntries={[path]}><AppRoutes client={client} /></MemoryRouter></StrictMode>)
}

async function fill(password = 'password123') {
  const user = userEvent.setup()
  await user.type(await screen.findByLabelText('이메일'), person.email)
  await user.type(screen.getByLabelText('비밀번호'), password)
  return user
}

describe('authentication screens and routes', () => {
  // 확인: 예시 모드에서는 '예시 모드' 안내가 보이고 두 방향의 예시 문장이 번역되며(번역문에 lang), 네트워크
  // 요청은 로그인 복원 두 번뿐이다(번역은 서버로 가지 않음).
  it('connects /translate to the contract demo and supports both translation directions', async () => {
    vi.stubEnv('VITE_TRANSLATION_MOCK', 'true')
    sessionStorage.setItem(REFRESH_KEY, 'refresh')
    fetchMock.mockResolvedValueOnce(response(tokens)).mockResolvedValueOnce(response(person))
    show('/translate')
    expect(await screen.findByText('예시 모드')).toBeInTheDocument()
    const user = userEvent.setup()
    fireEvent.change(screen.getByLabelText(/번역할 글/), { target: { value: '안녕하세요' } })
    await user.click(screen.getByRole('button', { name: '번역하기' }))
    expect(await screen.findByText('Hello')).toHaveAttribute('lang', 'en')
    await user.selectOptions(screen.getByLabelText('말하거나 입력할 언어'), 'en')
    fireEvent.change(screen.getByLabelText(/번역할 글/), { target: { value: 'Thank you' } })
    await user.click(screen.getByRole('button', { name: '번역하기' }))
    expect(await screen.findByText('감사합니다')).toHaveAttribute('lang', 'ko')
    expect(fetchMock).toHaveBeenCalledTimes(2) // Only real authentication, no demo network request.
  })

  // 확인: 예시 모드를 끄면 실제 /api/translate/text를 부르고, 401 뒤 갱신도 거절되면 refresh 토큰을 지우고
  // 로그인 화면으로 간다.
  it('uses the actual translation endpoint when the demo is disabled and returns to login on expired auth', async () => {
    vi.stubEnv('VITE_TRANSLATION_MOCK', 'false')
    sessionStorage.setItem(REFRESH_KEY, 'refresh')
    fetchMock.mockResolvedValueOnce(response(tokens)).mockResolvedValueOnce(response(person))
    show('/translate')
    await screen.findByRole('heading', { name: '번역' })
    expect(screen.queryByText('예시 모드')).not.toBeInTheDocument()
    fetchMock.mockResolvedValueOnce(response({}, 401)).mockResolvedValueOnce(response({}, 401))
    fireEvent.change(screen.getByLabelText(/번역할 글/), { target: { value: '안녕하세요' } })
    await userEvent.click(screen.getByRole('button', { name: '번역하기' }))
    expect(await screen.findByRole('heading', { name: '로그인' })).toBeInTheDocument()
    expect(fetchMock.mock.calls[2][0]).toBe('/api/translate/text')
    expect(fetchMock.mock.calls[3][0]).toBe('/api/auth/refresh')
    expect(sessionStorage.length).toBe(0)
  })

  // 확인: 로그인하지 않고 /history로 오면 로그인 화면으로 가고, 로그인하면 원래 가려던 기록 화면으로
  // 돌아간다. 메뉴로 번역 화면에 갈 수 있고, 로그아웃하면 저장소가 비고 로그인 화면이 된다.
  it('protects a direct history link and returns there after login, then logs out', async () => {
    fetchMock.mockResolvedValueOnce(response(tokens)).mockResolvedValueOnce(response(person)).mockImplementation(async () => response({ items: [], total: 0 }))
    show('/history')
    expect(await screen.findByRole('heading', { name: '로그인' })).toBeInTheDocument()
    const user = await fill()
    await user.click(screen.getByRole('button', { name: '로그인' }))
    expect(await screen.findByRole('heading', { name: '번역 기록' })).toBeInTheDocument()
    await user.click(screen.getByRole('link', { name: '번역' }))
    expect(await screen.findByRole('heading', { name: '번역' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '로그아웃' }))
    expect(await screen.findByRole('heading', { name: '로그인' })).toBeInTheDocument()
    expect(sessionStorage.length).toBe(0)
    expect(localStorage.length).toBe(0)
  })

  // 확인: 가입은 JSON으로 이메일·비밀번호를 한 번 보내고(한글 24자 = 72바이트 경계 통과), 토큰을 저장하지
  // 않은 채 '가입 완료' 안내와 함께 로그인 화면으로 가며 비밀번호 입력란을 비운다.
  it('registers with JSON then asks the user to log in', async () => {
    fetchMock.mockResolvedValueOnce(response(person, 201))
    show('/register')
    const user = await fill('가'.repeat(24))
    await user.click(screen.getByRole('button', { name: '가입하기' }))
    expect(await screen.findByRole('heading', { name: '로그인' })).toBeInTheDocument()
    expect(screen.getByRole('status')).toHaveTextContent('가입이 완료')
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock.mock.calls[0][0]).toBe('/api/auth/register')
    expect(JSON.parse(fetchMock.mock.calls[0][1]?.body as string)).toEqual({ email: person.email, password: '가'.repeat(24) })
    expect(sessionStorage.length).toBe(0)
    expect(screen.getByLabelText('비밀번호')).toHaveValue('')
  })

  // 확인: 8자 미만이거나 UTF-8 72바이트를 넘는(한글 25자 = 75바이트) 비밀번호는 요청 없이 화면에서 막는다.
  it.each(['short', '가'.repeat(25)])('validates password character and byte limits before registration: %s', async (password) => {
    show('/register')
    const user = await fill(password)
    await user.click(screen.getByRole('button', { name: '가입하기' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('72바이트')
    expect(fetchMock).not.toHaveBeenCalled()
  })

  // 확인: 가입의 409(이미 가입)·422(조건 불일치)는 고정 안내로 보이고, 서버 응답에 담긴 입력값(비밀번호)은
  // 화면에 나오지 않는다.
  it.each([409, 422])('shows registration errors for HTTP %s without exposing response input', async (status) => {
    fetchMock.mockResolvedValueOnce(response({ detail: [{ input: 'secret-password' }] }, status))
    show('/register')
    const user = await fill()
    await user.click(screen.getByRole('button', { name: '가입하기' }))
    expect(await screen.findByRole('alert')).toHaveTextContent(status === 409 ? '이미 가입된 이메일' : '조건을 확인')
    expect(screen.queryByText('secret-password')).not.toBeInTheDocument()
  })

  // 확인: 틀린 로그인은 안내를 보이고 버튼을 다시 켜며 비밀번호 입력란을 비운다. 토큰 갱신을 시도하지
  // 않는다(요청 1번).
  it('shows invalid credentials and allows a retry', async () => {
    fetchMock.mockResolvedValueOnce(response({}, 401))
    show()
    const user = await fill()
    await user.click(screen.getByRole('button', { name: '로그인' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('이메일 또는 비밀번호')
    expect(screen.getByRole('button', { name: '로그인' })).toBeEnabled()
    expect(screen.getByLabelText('비밀번호')).toHaveValue('')
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  // 확인: fetch가 네트워크 오류로 실패하면 '서버에 연결할 수 없습니다' 안내를 보인다.
  it('shows a network error', async () => {
    fetchMock.mockRejectedValueOnce(new TypeError('Failed to fetch'))
    show()
    const user = await fill()
    await user.click(screen.getByRole('button', { name: '로그인' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('서버에 연결할 수 없습니다')
  })

  // 확인: refresh 토큰이 있으면 /history 직접 링크에서 세션을 복원해 기록 화면을 보이고, StrictMode에서도
  // 갱신과 사용자 확인 요청이 한 번씩만 나간다.
  it('restores the session on a direct link under StrictMode', async () => {
    sessionStorage.setItem(REFRESH_KEY, 'old-refresh')
    fetchMock.mockResolvedValueOnce(response(tokens)).mockResolvedValueOnce(response(person)).mockImplementation(async () => response({ items: [], total: 0 }))
    show('/history')
    expect(await screen.findByRole('heading', { name: '번역 기록' })).toBeInTheDocument()
    expect(await screen.findByText(/아직 번역 기록이 없습니다/)).toBeInTheDocument()
    expect(fetchMock.mock.calls.filter(([path]) => path === '/api/auth/refresh')).toHaveLength(1)
    expect(fetchMock.mock.calls.filter(([path]) => path === '/api/auth/me')).toHaveLength(1)
  })

  // 확인: 화면을 보는 중 요청이 401이고 갱신도 거절되면, 로그인 화면으로 가서 '로그인이 만료' 안내를 띄우고
  // 토큰을 지운다.
  it('returns to login when an authenticated request cannot refresh', async () => {
    sessionStorage.setItem(REFRESH_KEY, 'old-refresh')
    fetchMock.mockResolvedValueOnce(response(tokens)).mockResolvedValueOnce(response(person))
    show('/translate')
    await screen.findByRole('heading', { name: '번역' })
    fetchMock.mockResolvedValueOnce(response({}, 401)).mockResolvedValueOnce(response({}, 401))
    await act(async () => { await client.request('/api/history').catch(() => undefined) })
    expect(await screen.findByRole('heading', { name: '로그인' })).toBeInTheDocument()
    expect(screen.getByRole('status')).toHaveTextContent('로그인이 만료')
    expect(sessionStorage.length).toBe(0)
  })

  // 확인: 세션 복원이 거절되면 보호된 기록 화면을 한순간도 그리지 않고 로그인 화면만 보인다.
  it('keeps protected content hidden if restoration fails', async () => {
    sessionStorage.setItem(REFRESH_KEY, 'expired-refresh')
    fetchMock.mockResolvedValueOnce(response({}, 401))
    show('/history')
    expect(await screen.findByRole('heading', { name: '로그인' })).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: '번역 기록' })).not.toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })
})

// 확인: VITE_TRANSLATION_MOCK이 없거나 'false'거나 대문자 'TRUE'면(정확히 'true'가 아니면) 개발 중에도 실제
// API를 쓴다. 모델 없이 뜬 서버의 503은 고정 안내로 보이고, 서버 응답의 문자열·입력값은 화면에 나오지 않는다.
it.each([undefined, 'false', 'TRUE'])('defaults to the real API in development for mock=%s and displays a model-free server 503 safely', async (mock) => {
  vi.stubEnv('DEV', true)
  vi.stubEnv('VITE_TRANSLATION_MOCK', mock)
  sessionStorage.setItem(REFRESH_KEY, 'refresh')
  fetchMock.mockResolvedValueOnce(response(tokens)).mockResolvedValueOnce(response(person))
    .mockResolvedValueOnce(response({ detail: 'Translation service is unavailable', input: 'private-server-input' }, 503))
  show('/translate')
  await screen.findByRole('heading', { name: '번역' })
  expect(screen.queryByText('예시 모드')).not.toBeInTheDocument()
  fireEvent.change(screen.getByLabelText(/번역할 글/), { target: { value: '안녕하세요' } })
  await userEvent.click(screen.getByRole('button', { name: '번역하기' }))
  expect(await screen.findByRole('alert')).toHaveTextContent('번역 서비스를 사용할 수 없습니다. 잠시 후 다시 시도해 주세요.')
  expect(screen.queryByText(/private-server-input|Translation service is unavailable/)).not.toBeInTheDocument()
  expect(fetchMock.mock.calls[2][0]).toBe('/api/translate/text')
  expect(new Headers(fetchMock.mock.calls[2][1]?.headers).get('Authorization')).toBe('Bearer access')
})

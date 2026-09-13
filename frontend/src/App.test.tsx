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
  sessionStorage.clear()
  localStorage.clear()
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
  client = new ApiClient()
})
afterEach(() => { vi.unstubAllGlobals(); vi.unstubAllEnvs() })

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

  it('protects a direct history link and returns there after login, then logs out', async () => {
    fetchMock.mockResolvedValueOnce(response(tokens)).mockResolvedValueOnce(response(person))
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

  it.each(['short', '가'.repeat(25)])('validates password character and byte limits before registration: %s', async (password) => {
    show('/register')
    const user = await fill(password)
    await user.click(screen.getByRole('button', { name: '가입하기' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('72바이트')
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it.each([409, 422])('shows registration errors for HTTP %s without exposing response input', async (status) => {
    fetchMock.mockResolvedValueOnce(response({ detail: [{ input: 'secret-password' }] }, status))
    show('/register')
    const user = await fill()
    await user.click(screen.getByRole('button', { name: '가입하기' }))
    expect(await screen.findByRole('alert')).toHaveTextContent(status === 409 ? '이미 가입된 이메일' : '조건을 확인')
    expect(screen.queryByText('secret-password')).not.toBeInTheDocument()
  })

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

  it('shows a network error', async () => {
    fetchMock.mockRejectedValueOnce(new TypeError('Failed to fetch'))
    show()
    const user = await fill()
    await user.click(screen.getByRole('button', { name: '로그인' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('서버에 연결할 수 없습니다')
  })

  it('restores the session on a direct link under StrictMode', async () => {
    sessionStorage.setItem(REFRESH_KEY, 'old-refresh')
    fetchMock.mockResolvedValueOnce(response(tokens)).mockResolvedValueOnce(response(person))
    show('/history')
    expect(await screen.findByRole('heading', { name: '번역 기록' })).toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

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

  it('keeps protected content hidden if restoration fails', async () => {
    sessionStorage.setItem(REFRESH_KEY, 'expired-refresh')
    fetchMock.mockResolvedValueOnce(response({}, 401))
    show('/history')
    expect(await screen.findByRole('heading', { name: '로그인' })).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: '번역 기록' })).not.toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })
})

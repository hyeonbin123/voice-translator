import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest'
import { ApiClient, ApiError, REFRESH_KEY } from './client'

const user = { id: 'user-1', email: 'person@example.com', created_at: '2026-09-13T00:00:00Z' }
const tokens = (suffix = '1') => ({ access_token: `access-${suffix}`, refresh_token: `refresh-${suffix}`, token_type: 'bearer', expires_in: 1800 })
const response = (body: unknown = {}, status = 200) => new Response(JSON.stringify(body), { status })
const fetchMock = vi.fn<typeof fetch>()
let client: ApiClient

beforeEach(() => {
  sessionStorage.clear()
  localStorage.clear()
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
  client = new ApiClient()
})
afterEach(() => vi.unstubAllGlobals())

async function login() {
  fetchMock.mockResolvedValueOnce(response(tokens())).mockResolvedValueOnce(response(user))
  await client.login(user.email, ' password+한글 ')
  fetchMock.mockClear()
}

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => { resolve = done })
  return { promise, resolve }
}

describe('API authentication', () => {
  it('sends the OAuth2 form, preserves passwords, and persists only the refresh token', async () => {
    fetchMock.mockResolvedValueOnce(response(tokens())).mockResolvedValueOnce(response(user))
    await client.login(user.email, ' password+한글 ')
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('/api/auth/login')
    expect(new Headers(init?.headers).get('Content-Type')).toBe('application/x-www-form-urlencoded')
    expect(new URLSearchParams(init?.body as URLSearchParams).get('username')).toBe(user.email)
    expect(new URLSearchParams(init?.body as URLSearchParams).get('password')).toBe(' password+한글 ')
    expect(new Headers(fetchMock.mock.calls[1][1]?.headers).get('Authorization')).toBe('Bearer access-1')
    expect(sessionStorage.length).toBe(1)
    expect(sessionStorage.getItem(REFRESH_KEY)).toBe('refresh-1')
    expect(localStorage.length).toBe(0)
    expect(client.getSnapshot().user).toEqual(user)
  })

  it('restores a fresh client with one refresh, even when initialized twice', async () => {
    sessionStorage.setItem(REFRESH_KEY, 'refresh-old')
    fetchMock.mockResolvedValueOnce(response(tokens('2'))).mockResolvedValueOnce(response(user))
    await Promise.all([client.initialize(), client.initialize()])
    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(fetchMock.mock.calls[0][0]).toBe('/api/auth/refresh')
    expect(new Headers(fetchMock.mock.calls[0][1]?.headers).has('Authorization')).toBe(false)
    expect(JSON.parse(fetchMock.mock.calls[0][1]?.body as string)).toEqual({ refresh_token: 'refresh-old' })
    expect(client.getSnapshot().status).toBe('authenticated')
    expect(sessionStorage.getItem(REFRESH_KEY)).toBe('refresh-2')
  })

  it('starts anonymously without a refresh token or network request', async () => {
    await client.initialize()
    expect(client.getSnapshot().status).toBe('anonymous')
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('refreshes one time on 401 and replays the original request with the new access token', async () => {
    await login()
    fetchMock.mockResolvedValueOnce(response({}, 401)).mockResolvedValueOnce(response(tokens('2'))).mockResolvedValueOnce(response({ ok: true }))
    await client.request('/api/translate/text', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{"text":"hello"}' })
    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual(['/api/translate/text', '/api/auth/refresh', '/api/translate/text'])
    const retry = fetchMock.mock.calls[2][1]
    expect(new Headers(retry?.headers).get('Authorization')).toBe('Bearer access-2')
    expect(retry?.method).toBe('POST')
    expect(retry?.body).toBe('{"text":"hello"}')
    expect(sessionStorage.getItem(REFRESH_KEY)).toBe('refresh-2')
  })

  it('shares a refresh for concurrent 401 responses', async () => {
    await login()
    const renewal = deferred<Response>()
    fetchMock.mockImplementation(async (url, init) => {
      if (url === '/api/auth/refresh') return renewal.promise
      return response({}, new Headers(init?.headers).get('Authorization') === 'Bearer access-1' ? 401 : 200)
    })
    const first = client.request('/api/history')
    const second = client.request('/api/auth/me')
    await vi.waitFor(() => expect(fetchMock.mock.calls.filter(([url]) => url === '/api/auth/refresh')).toHaveLength(1))
    renewal.resolve(response(tokens('2')))
    await Promise.all([first, second])
    expect(fetchMock.mock.calls.filter(([url]) => url === '/api/auth/refresh')).toHaveLength(1)
  })

  it('does not refresh again for a delayed 401 using an already replaced token', async () => {
    await login()
    const slow = deferred<Response>()
    fetchMock.mockReturnValueOnce(slow.promise).mockResolvedValueOnce(response({}, 401)).mockResolvedValueOnce(response(tokens('2'))).mockResolvedValue(response({}))
    const old = client.request('/api/history')
    await client.request('/api/auth/me')
    slow.resolve(response({}, 401))
    await old
    expect(fetchMock.mock.calls.filter(([url]) => url === '/api/auth/refresh')).toHaveLength(1)
  })

  it.each([401, 500])('clears the session when refresh fails with %s', async (status) => {
    await login()
    fetchMock.mockResolvedValueOnce(response({}, 401)).mockResolvedValueOnce(response({}, status))
    await expect(client.request('/api/history')).rejects.toBeInstanceOf(ApiError)
    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(client.getSnapshot()).toEqual({ status: 'anonymous', user: null, expired: true })
    expect(sessionStorage.getItem(REFRESH_KEY)).toBeNull()
  })

  it('stops after a retried request also returns 401', async () => {
    await login()
    fetchMock.mockResolvedValueOnce(response({}, 401)).mockResolvedValueOnce(response(tokens('2'))).mockResolvedValueOnce(response({}, 401))
    await expect(client.request('/api/history')).rejects.toBeInstanceOf(ApiError)
    expect(fetchMock).toHaveBeenCalledTimes(3)
    expect(client.getSnapshot().status).toBe('anonymous')
  })

  it('does not refresh on ordinary server errors', async () => {
    await login()
    fetchMock.mockResolvedValueOnce(response({}, 500))
    await expect(client.request('/api/history')).rejects.toBeInstanceOf(ApiError)
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(client.getSnapshot().status).toBe('authenticated')
  })

  it('does not retry failed login credentials with refresh', async () => {
    fetchMock.mockResolvedValueOnce(response({}, 401))
    await expect(client.login(user.email, 'wrong')).rejects.toBeInstanceOf(ApiError)
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(sessionStorage.length).toBe(0)
  })

  it('does not restore tokens from an in-flight refresh after logout', async () => {
    await login()
    const renewal = deferred<Response>()
    fetchMock.mockResolvedValueOnce(response({}, 401)).mockReturnValueOnce(renewal.promise)
    const pending = client.request('/api/history')
    const rejected = expect(pending).rejects.toThrow('로그인 상태가 바뀌었습니다')
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    client.logout()
    renewal.resolve(response(tokens('2')))
    await rejected
    expect(sessionStorage.length).toBe(0)
    expect(client.getSnapshot().status).toBe('anonymous')
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it('ignores an old login response after logout', async () => {
    const pendingLogin = deferred<Response>()
    fetchMock.mockReturnValueOnce(pendingLogin.promise)
    const pending = client.login(user.email, 'password')
    const rejected = expect(pending).rejects.toThrow('로그인 상태가 바뀌었습니다')
    client.logout()
    pendingLogin.resolve(response(tokens()))
    await rejected
    expect(sessionStorage.length).toBe(0)
  })

  it('returns blob responses and handles 204 without assuming JSON', async () => {
    await login()
    fetchMock.mockResolvedValueOnce(new Response('audio', { headers: { 'Content-Type': 'audio/wav' } })).mockResolvedValueOnce(new Response(null, { status: 204 }))
    expect((await (await client.request('/api/audio/id')).blob()).type).toBe('audio/wav')
    expect((await client.request('/api/history/id', { method: 'DELETE' })).status).toBe(204)
  })
})

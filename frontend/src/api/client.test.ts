import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest'
import { ApiClient, ApiError, REFRESH_KEY } from './client'

// [파일 역할] API 클라이언트(api/client.ts) 테스트 묶음. 화면 없이 ApiClient만 만들어 로그인·세션 복원·토큰
// 갱신·로그아웃의 경쟁 상태와, 서버 오류를 안내 문구로 바꾸는 규칙을 확인한다.
// [무엇을 지키나] (1) 토큰 보관: refresh만 sessionStorage에, localStorage에는 아무것도 없음. (2) 401이면 한
// 번만 갱신하고, 동시에 와도 갱신은 하나. (3) 갱신이 거절(401·422)될 때만 로그아웃하고, 5xx·네트워크 오류는
// 토큰을 지키기(0db7b87). (4) 로그아웃 뒤 늦게 온 응답으로 세션이 되살아나지 않기. (5) 계약에 있는 상태
// 코드·detail·경로 조합만 고유 안내로 바꾸기.
// [가짜 객체] fetch를 가짜로 바꿔 응답과 그 시점을 정한다. deferred로 응답을 붙잡아 두면 '요청 중 로그아웃'
// 같은 순서를 재현할 수 있다.

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

// 로그인한 상태를 만드는 도우미(토큰 응답 + 사용자 응답). 비밀번호에 앞뒤 공백·기호·한글을 넣어 그대로
// 보내는지도 함께 쓴다. 끝나면 fetch 기록을 지운다.
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
  // 확인: 웹소켓용 토큰(liveToken)은 메모리 토큰이 있으면 요청 없이 주고, 거절된 토큰을 넘기면 갱신하되
  // 동시에 두 번 불러도 갱신은 한 번이며, 이미 바뀐 뒤에는 다시 갱신하지 않는다. 갱신 중 로그아웃하면 늦게 온
  // 토큰을 저장하지 않고 '로그인 상태가 바뀌었습니다'로 실패한다.
  it('shares refresh for live authentication and never restores a logged-out session', async () => {
    await login()
    expect(await client.liveToken()).toBe('access-1')
    expect(fetchMock).not.toHaveBeenCalled()
    fetchMock.mockResolvedValueOnce(response(tokens('2')))
    expect(await Promise.all([client.liveToken('access-1'), client.liveToken('access-1')])).toEqual(['access-2', 'access-2'])
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(await client.liveToken('access-1')).toBe('access-2')
    expect(fetchMock).toHaveBeenCalledTimes(1)
    const pending = deferred<Response>(); fetchMock.mockReturnValueOnce(pending.promise)
    const refreshing = client.liveToken('access-2')
    client.logout(); pending.resolve(response(tokens('late')))
    await expect(refreshing).rejects.toThrow('로그인 상태가 바뀌었습니다')
    expect(client.getSnapshot().status).toBe('anonymous')
    expect(sessionStorage.getItem(REFRESH_KEY)).toBeNull()
  })

  // 확인: 로그인은 form-urlencoded로 username(이메일)과 비밀번호를 보내며 비밀번호를 자르거나 바꾸지
  // 않는다(docs/api.md: 공백을 자르거나 정규화하지 않음). 이어지는 /api/auth/me에 새 access 토큰이 붙고,
  // 저장소에는 refresh 토큰 하나만 남는다.
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

  // 확인: 새로 고침처럼 새 클라이언트가 저장된 refresh 토큰으로 세션을 복원한다. initialize를 두 번 불러도
  // 갱신은 한 번이며, 갱신 요청에는 Authorization이 없고 본문에 refresh 토큰만 담는다. 받은 새 refresh
  // 토큰으로 바꿔 저장한다.
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

  // 확인: 저장된 토큰이 없으면 네트워크 요청 없이 바로 로그아웃 상태로 시작한다.
  it('starts anonymously without a refresh token or network request', async () => {
    await client.initialize()
    expect(client.getSnapshot().status).toBe('anonymous')
    expect(fetchMock).not.toHaveBeenCalled()
  })

  // 확인: 401을 받으면 갱신하고 원래 요청을 같은 메서드·본문으로 새 토큰과 함께 다시 보낸다.
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

  // 확인: 두 요청이 동시에 401을 받아도 갱신 요청은 하나만 나가고 둘 다 그 결과를 기다린다.
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

  // 확인: 옛 토큰으로 보낸 느린 요청의 401이 이미 토큰이 바뀐 뒤에 오면, 다시 갱신하지 않고 새 토큰으로 바로
  // 다시 보낸다.
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

  // 확인: 갱신이 401·422로 거절되면 세션을 끝내고(expired: true로 만료 안내) refresh 토큰을 지운다.
  it.each([401, 422])('clears the session when refresh is rejected with %s', async (status) => {
    await login()
    fetchMock.mockResolvedValueOnce(response({}, 401)).mockResolvedValueOnce(response({}, status))
    await expect(client.request('/api/history')).rejects.toBeInstanceOf(ApiError)
    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(client.getSnapshot()).toEqual({ status: 'anonymous', user: null, expired: true })
    expect(sessionStorage.getItem(REFRESH_KEY)).toBeNull()
  })

  // 확인: 앱 시작 때 저장된 토큰의 갱신이 거절되어도 같은 처리(만료 안내, 토큰 삭제)를 한다.
  it.each([401, 422])('ends a stored session whose refresh is rejected with %s at startup', async (status) => {
    sessionStorage.setItem(REFRESH_KEY, 'refresh-old')
    fetchMock.mockResolvedValueOnce(response({}, status))
    await client.initialize()
    expect(client.getSnapshot()).toEqual({ status: 'anonymous', user: null, expired: true })
    expect(sessionStorage.getItem(REFRESH_KEY)).toBeNull()
  })

  // 일시 오류 세 가지: 500, nginx의 HTML 502 페이지, 네트워크 끊김(fetch의 TypeError). API가 다시 시작하는
  // 동안 프록시가 내는 5xx나 끊긴 연결은 갱신 토큰이 무효라는 뜻이 아니다.
  const transient = [
    ['500', () => Promise.resolve(response({}, 500))],
    ['502', () => Promise.resolve(new Response('<html>502 Bad Gateway</html>', { status: 502 }))],
    ['network', () => Promise.reject(new TypeError('Failed to fetch'))],
  ] as const

  // 확인: 사용 중 갱신이 일시 오류로 실패하면 그 요청만 실패하고 세션과 refresh 토큰은 남으며, 다음 401에서
  // 다시 갱신해 성공한다.
  it.each(transient)('keeps the session when refresh fails with %s and refreshes on the next 401', async (_name, reply) => {
    await login()
    fetchMock.mockResolvedValueOnce(response({}, 401)).mockImplementationOnce(reply)
    await expect(client.request('/api/history')).rejects.toThrow()
    expect(client.getSnapshot()).toEqual({ status: 'authenticated', user, expired: false })
    expect(sessionStorage.getItem(REFRESH_KEY)).toBe('refresh-1')
    fetchMock.mockResolvedValueOnce(response({}, 401)).mockResolvedValueOnce(response(tokens('2'))).mockResolvedValueOnce(response({ ok: true }))
    expect((await client.request('/api/history')).status).toBe(200)
    expect(sessionStorage.getItem(REFRESH_KEY)).toBe('refresh-2')
  })

  // 확인: 시작 때 갱신이 일시 오류로 실패하면 만료 안내 없이 로그아웃 화면을 보이되 토큰은 남겨, 서버가
  // 돌아온 뒤 새로 고침(새 클라이언트)하면 세션이 복원된다.
  it.each(transient)('keeps the refresh token when refresh fails with %s at startup', async (_name, reply) => {
    sessionStorage.setItem(REFRESH_KEY, 'refresh-old')
    fetchMock.mockImplementationOnce(reply)
    await client.initialize()
    expect(client.getSnapshot()).toEqual({ status: 'anonymous', user: null, expired: false })
    expect(sessionStorage.getItem(REFRESH_KEY)).toBe('refresh-old')
    fetchMock.mockResolvedValueOnce(response(tokens('2'))).mockResolvedValueOnce(response(user))
    const reloaded = new ApiClient()
    await reloaded.initialize()
    expect(reloaded.getSnapshot()).toEqual({ status: 'authenticated', user, expired: false })
    expect(sessionStorage.getItem(REFRESH_KEY)).toBe('refresh-2')
  })

  // 확인: 갱신한 토큰으로 다시 보낸 요청도 401이면 더 반복하지 않고 로그아웃한다(요청 3번에서 멈춤).
  it('stops after a retried request also returns 401', async () => {
    await login()
    fetchMock.mockResolvedValueOnce(response({}, 401)).mockResolvedValueOnce(response(tokens('2'))).mockResolvedValueOnce(response({}, 401))
    await expect(client.request('/api/history')).rejects.toBeInstanceOf(ApiError)
    expect(fetchMock).toHaveBeenCalledTimes(3)
    expect(client.getSnapshot().status).toBe('anonymous')
  })

  // 확인: 500 같은 일반 서버 오류에는 토큰을 갱신하지 않고 세션도 그대로 둔다.
  it('does not refresh on ordinary server errors', async () => {
    await login()
    fetchMock.mockResolvedValueOnce(response({}, 500))
    await expect(client.request('/api/history')).rejects.toBeInstanceOf(ApiError)
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(client.getSnapshot().status).toBe('authenticated')
  })

  // 확인: 로그인 자체의 401(틀린 비밀번호)은 갱신 대상이 아니며 아무것도 저장하지 않는다.
  it('does not retry failed login credentials with refresh', async () => {
    fetchMock.mockResolvedValueOnce(response({}, 401))
    await expect(client.login(user.email, 'wrong')).rejects.toBeInstanceOf(ApiError)
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(sessionStorage.length).toBe(0)
  })

  // 확인: 401 뒤 갱신이 진행되는 중에 로그아웃하면, 늦게 도착한 새 토큰을 저장하지 않고 원래 요청도 다시
  // 보내지 않는다.
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

  // 확인: 로그인 응답을 기다리는 중에 로그아웃하면, 늦게 온 로그인 응답의 토큰을 저장하지 않는다.
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

  // 확인: request는 응답 본문을 해석하지 않고 Response를 그대로 돌려줘 음성(Blob)과 204 빈 응답도 다룰 수
  // 있다.
  it('returns blob responses and handles 204 without assuming JSON', async () => {
    await login()
    fetchMock.mockResolvedValueOnce(new Response('audio', { headers: { 'Content-Type': 'audio/wav' } })).mockResolvedValueOnce(new Response(null, { status: 204 }))
    expect((await (await client.request('/api/audio/id')).blob()).type).toBe('audio/wav')
    expect((await client.request('/api/history/id', { method: 'DELETE' })).status).toBe(204)
  })
})

describe('translation contract errors', () => {
  // 확인: 번역·음성 경로의 상태 코드와 detail이 계약(docs/api.md '실패했을 때' 표)과 정확히 맞을 때만 그
  // 안내를 쓰고, 상태 코드가 다르거나(503 + 'No speech…'), 계약 밖 detail이거나, 다른 경로(/api/history의
  // 'Audio not found')면 일반 문구로 떨어진다. 응답의 입력값은 문구에 들어가지 않는다.
  it.each([
    [413, 'Audio file is larger than 10 MB', '음성 파일은 10MB 이하로 선택해 주세요.', '/api/translate/speech'],
    [422, 'Audio could not be decoded', '음성 파일을 읽을 수 없습니다. 다시 녹음하거나 다른 파일을 선택해 주세요.', '/api/translate/speech'],
    [422, 'Audio is longer than 30 seconds', '음성은 30초 이하로 녹음해 주세요.', '/api/translate/speech'],
    [422, 'No speech was recognized', '말소리를 찾지 못했습니다. 마이크를 확인하고 다시 녹음해 주세요.', '/api/translate/speech'],
    [503, 'Translation service is unavailable', '번역 서비스를 사용할 수 없습니다. 잠시 후 다시 시도해 주세요.', '/api/translate/text'],
    [404, 'Audio not found', '번역 음성이 없거나 삭제되었습니다. 다시 번역해 주세요.', '/api/audio/id'],
    [503, 'unknown private input', '요청을 처리하지 못했습니다. 잠시 후 다시 시도해 주세요.', '/api/translate/text'],
    [503, 'No speech was recognized', '요청을 처리하지 못했습니다. 잠시 후 다시 시도해 주세요.', '/api/translate/speech'],
    [422, [{ input: 'unknown private input', msg: 'unknown private input' }], '입력 내용과 번역 방향을 확인해 주세요.', '/api/translate/text'],
    [404, 'Audio not found', '요청을 처리하지 못했습니다. 잠시 후 다시 시도해 주세요.', '/api/history/id'],
  ] as const)('maps only the allowed status/detail/path for %s %j', async (status, detail, message, path) => {
    await login()
    fetchMock.mockResolvedValueOnce(response({ detail, input: 'unknown private input' }, status))
    await expect(client.request(path)).rejects.toMatchObject({ name: 'ApiError', status, message })
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  // 확인: nginx가 JSON이 아닌 HTML 오류 페이지(예: 413)를 돌려줘도 본문을 읽다 실패하지 않고 일반 문구를
  // 쓴다.
  it('uses safe local text for an HTML proxy error', async () => {
    await login()
    fetchMock.mockResolvedValueOnce(new Response('<html>private input</html>', { status: 413 }))
    await expect(client.request('/api/translate/speech')).rejects.toMatchObject({
      status: 413, message: '요청을 처리하지 못했습니다. 잠시 후 다시 시도해 주세요.',
    })
  })
})

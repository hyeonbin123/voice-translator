// [파일 역할] 화면이 서버 API를 부르는 공통 창구. 로그인·가입·토큰 갱신·세션 복원을 맡고, 인증이 필요한
// 요청에 access 토큰을 붙이며, 서버 오류를 화면에 보일 고정 한국어 문구로 바꾼다.
// [설계 이유] 토큰 보관은 docs/api.md 인증 절을 따른다: access 토큰은 메모리에만, refresh 토큰은
// sessionStorage에 둔다(탭을 닫으면 사라짐). 401을 받으면 한 번만 갱신해 다시 보내고, 동시에 여러 요청이
// 401을 받아도 갱신 요청은 하나를 함께 기다린다. 갱신이 401·422로 거절될 때만 로그아웃하고, 서버·네트워크
// 오류(5xx, 연결 실패)면 refresh 토큰을 지우지 않아 다음 요청이나 새로 고침 때 다시 시도한다(0db7b87).
// 로그아웃·재로그인 뒤에 도착한 옛 응답은 세대 번호(generation)로 걸러 버린다(65c4044).
// [연결] 앱 전체가 맨 아래의 api 인스턴스 하나를 쓴다. App.tsx·AuthPage.tsx가 세션 상태를 구독하고,
// translation/api.ts·history/api.ts가 request로 번역·기록 API를 부르며, 웹소켓(live/socket.ts)은
// liveToken으로 토큰만 받는다.

// 로그인한 사용자 정보. GET /api/auth/me와 가입 응답의 모양이다.
export interface User {
  id: string
  email: string
  created_at: string
}

// 로그인·갱신 응답. 화면은 두 토큰만 쓰고 expires_in(초)은 보지 않는다. 만료는 서버의 401로 알게 된다.
interface Tokens {
  access_token: string
  refresh_token: string
  token_type: string
  expires_in: number
}

// 화면이 구독하는 로그인 상태. loading은 첫 세션 복원(initialize)이 끝나기 전, expired는 갱신이 거절돼 세션이
// 끝났음을 뜻하며 로그인 화면이 '로그인이 만료되었습니다' 안내를 띄우는 데 쓴다.
export interface Session {
  status: 'loading' | 'anonymous' | 'authenticated'
  user: User | null
  expired: boolean
}

// refresh 토큰을 넣어 두는 sessionStorage 키. 새로 고침해도 같은 탭이면 남아 세션을 되살릴 수 있다. 테스트도
// 이 키로 저장소를 미리 채운다.
export const REFRESH_KEY = 'voice-translator.refresh'

// 서버가 실패로 답한 요청. status는 HTTP 상태 코드, message는 화면에 그대로 보여도 되는 한국어 문구다(서버
// 응답 내용을 담지 않음).
export class ApiError extends Error {
  readonly status: number
  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

// 요청이 진행되는 동안 로그아웃되거나 다른 계정으로 로그인해 세션이 바뀐 경우. 그 응답은 이미 의미가 없으므로
// 버리고 다시 시도하라고 안내한다.
class SessionChangedError extends Error {
  constructor() { super('로그인 상태가 바뀌었습니다. 다시 시도해 주세요.') }
}

// 응답이 성공이면 그대로 돌려주고, 실패면 ApiError로 바꿔 던진다.
// path로 문구 묶음을 고른다: /api/auth/ 경로는 로그인·가입 문구, 나머지는 일반 문구. 기본값이 로그인 경로인
// 이유는 로그인·가입·갱신 호출이 path 없이 부르기 때문이다.
// 번역·음성 경로는 응답 본문의 detail까지 보고 docs/api.md '실패했을 때' 표의 고정 문자열과 맞으면 그에 맞는
// 안내를 쓴다. 이 문자열은 서버와 화면 사이의 계약이라 한쪽만 바꾸면 안내가 일반 문구로 떨어진다.
// 서버의 422 응답에는 입력값(input)이 들어 있을 수 있어, 응답 내용은 화면에 절대 그대로 옮기지
// 않는다(App.test.tsx가 가입 비밀번호가 화면에 나오지 않는지 확인).
async function checked(response: Response, path = '/api/auth/login'): Promise<Response> {
  if (response.ok) return response
  // 알려진 상태 코드·detail 쌍일 때만 화면 문구를 고른다. 응답에 담긴 입력값은 절대 그대로 보여 주지 않는다.
  const auth = path.startsWith('/api/auth/')
  const messages: Record<number, string> = auth ? {
    401: '이메일 또는 비밀번호를 확인해 주세요.',
    409: '이미 가입된 이메일입니다. 로그인해 주세요.',
    422: '입력한 이메일과 비밀번호 조건을 확인해 주세요.',
    429: '요청이 많습니다. 잠시 후 다시 시도해 주세요.',
  } : {
    401: '로그인이 만료되었습니다. 다시 로그인해 주세요.',
    422: '입력 내용과 번역 방향을 확인해 주세요.',
    429: '요청이 많습니다. 잠시 후 다시 시도해 주세요.',
  }
  // '상태코드:detail' → 안내 문구. 413 파일 크기, 422 디코딩 실패·30초 초과·말소리 없음, 503 모델 없음, 404
  // 음성 없음(docs/api.md).
  const translationErrors: Record<string, string> = {
    '413:Audio file is larger than 10 MB': '음성 파일은 10MB 이하로 선택해 주세요.',
    '422:Audio could not be decoded': '음성 파일을 읽을 수 없습니다. 다시 녹음하거나 다른 파일을 선택해 주세요.',
    '422:Audio is longer than 30 seconds': '음성은 30초 이하로 녹음해 주세요.',
    '422:No speech was recognized': '말소리를 찾지 못했습니다. 마이크를 확인하고 다시 녹음해 주세요.',
    '503:Translation service is unavailable': '번역 서비스를 사용할 수 없습니다. 잠시 후 다시 시도해 주세요.',
    '404:Audio not found': '번역 음성이 없거나 삭제되었습니다. 다시 번역해 주세요.',
  }
  // 본문이 JSON이 아니어도(nginx의 HTML 오류 페이지 등) 실패하지 않게 catch로 null을 받고, detail이 문자열일
  // 때만 표에서 찾는다. 표에 없으면 아래 일반 문구로 넘어간다.
  if (path.startsWith('/api/translate/') || path.startsWith('/api/audio/')) {
    const body: unknown = await response.json().catch(() => null)
    const detail = body && typeof body === 'object' && 'detail' in body ? body.detail : null
    const message = typeof detail === 'string' ? translationErrors[`${response.status}:${detail}`] : undefined
    if (message) throw new ApiError(response.status, message)
  }
  throw new ApiError(response.status, messages[response.status] ?? '요청을 처리하지 못했습니다. 잠시 후 다시 시도해 주세요.')
}

// 화면이 오류를 보여 줄 때 쓰는 단일 함수. 우리가 만든 오류(ApiError, SessionChangedError)는 그 문구를, 그
// 밖의 오류(fetch가 던지는 TypeError 같은 네트워크 실패)는 연결 확인 문구를 돌려준다. 예상하지 못한 오류
// 메시지가 화면에 새지 않는다.
export function errorMessage(error: unknown): string {
  if (error instanceof ApiError || error instanceof SessionChangedError) return error.message
  return '서버에 연결할 수 없습니다. 연결을 확인하고 다시 시도해 주세요.'
}

// 인증 정보가 거절됐을 때(401·422)만 세션을 끝낸다. 프록시 5xx나 네트워크 오류에서는 갱신 토큰을 남겨 두어
// 다음 요청이나 새로 고침에서 다시 시도할 수 있게 한다(서버가 토큰을 폐기할 수 없으므로 지워도 얻는 것이 없다).
const authRejected = (error: unknown) => error instanceof ApiError && (error.status === 401 || error.status === 422)

// 로그인 상태와 토큰을 가진 클라이언트.
// - accessToken: 메모리에만 둔다. 새로 고침하면 사라지고 refresh 토큰으로 다시 받는다
// - generation: 로그아웃(로그인 시작 포함)마다 1 늘어난다. 요청 시작 때 값을 기억했다가 응답 뒤 달라졌으면
//   결과를 버린다
// - refreshing·initializing: 진행 중인 갱신·복원 약속(Promise)을 기억해, 동시에 불려도 서버 요청은 한 번만
//   나가게 한다
// - listeners·session: useSyncExternalStore(auth/useSession.ts)용 구독자 목록과 현재 상태
export class ApiClient {
  private accessToken: string | null = null
  private generation = 0
  private refreshing: Promise<void> | null = null
  private initializing: Promise<void> | null = null
  private listeners = new Set<() => void>()
  private session: Session = { status: 'loading', user: null, expired: false }
  private readonly storage: Storage

  // 저장소를 바꿔 끼울 수 있게 해 테스트가 가짜 저장소를 넣을 수 있다. 기본은 sessionStorage.
  constructor(storage: Storage = window.sessionStorage) { this.storage = storage }

  // useSyncExternalStore가 부르는 두 함수. 화살표 함수 필드라 this가 고정되고 늘 같은 함수다. getSnapshot은
  // 상태가 바뀔 때만 새 객체를 돌려줘야 React가 불필요하게 다시 그리지 않는다(publish가 객체를 통째로
  // 바꾼다).
  getSnapshot = (): Session => this.session
  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener)
    return () => { this.listeners.delete(listener) }
  }

  // 새 상태를 저장하고 구독자(화면)에게 알린다.
  private publish(session: Session) {
    this.session = session
    this.listeners.forEach((listener) => listener())
  }

  // 요청을 시작할 때의 세대와 지금 세대가 다르면(그 사이 로그아웃·재로그인) SessionChangedError를 던져 옛
  // 응답을 쓰지 않게 한다. await 뒤마다 부른다.
  private assertCurrent(generation: number) {
    if (generation !== this.generation) throw new SessionChangedError()
  }

  // refresh는 sessionStorage에, access는 메모리에 둔다.
  private saveTokens(tokens: Tokens) {
    this.storage.setItem(REFRESH_KEY, tokens.refresh_token)
    this.accessToken = tokens.access_token
  }

  // 세션을 끝낸다. 세대를 올려 진행 중인 요청 결과를 무효로 만들고, 토큰과 진행 중인 갱신을 지운 뒤
  // anonymous를 알린다. expired=true면 로그인 화면이 만료 안내를 띄운다. 서버에는 토큰 폐기 기능이 없어
  // 로그아웃은 화면에서만 처리한다(docs/api.md).
  // 호출 위치: App.tsx의 로그아웃 버튼, 아래 refresh·request·initialize·login, translation/api.ts.
  logout = (expired = false) => {
    this.generation += 1
    this.accessToken = null
    this.refreshing = null
    this.storage.removeItem(REFRESH_KEY)
    this.publish({ status: 'anonymous', user: null, expired })
  }

  // refresh 토큰으로 새 토큰 쌍을 받는다. 이미 갱신 중이면 같은 약속을 돌려줘 여러 요청이 401을 받아도 갱신은
  // 한 번만 한다.
  // 저장된 refresh 토큰이 없으면 401과 같게 취급한다. 성공해도 그 사이 세션이 바뀌었으면 토큰을 저장하지
  // 않는다.
  // 실패 처리: 401·422(토큰 거절)이고 세션이 그대로일 때만 만료 로그아웃한다. 5xx·네트워크 오류는 토큰을 남겨
  // 두고 오류만 던진다. API 컨테이너가 다시 뜨며 모델을 올리는 동안의 502 같은 일시 오류로 7일짜리 로그인을
  // 잃지 않게 하려는 것이다(0db7b87).
  private refresh(): Promise<void> {
    if (this.refreshing) return this.refreshing
    const generation = this.generation
    const pending = (async () => {
      try {
        const refreshToken = this.storage.getItem(REFRESH_KEY)
        if (!refreshToken) throw new ApiError(401, '다시 로그인해 주세요.')
        const response = await checked(await fetch('/api/auth/refresh', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ refresh_token: refreshToken }), cache: 'no-store',
        }))
        const tokens: Tokens = await response.json()
        this.assertCurrent(generation)
        this.saveTokens(tokens)
      } catch (error) {
        if (generation === this.generation && authRejected(error)) this.logout(true)
        throw error
      }
    })()
    this.refreshing = pending
    // 예전 요청이 끝나면서 더 새로운 세션의 갱신 작업을 지우면 안 된다.
    void pending.then(() => {
      if (this.refreshing === pending) this.refreshing = null
    }, () => {
      if (this.refreshing === pending) this.refreshing = null
    })
    return pending
  }

  // liveToken: 웹소켓(WS /api/translate/live)의 첫 메시지에 넣을 access 토큰을 준다. 브라우저 WebSocket은
  // 인증 헤더를 붙일 수 없고 주소에 넣은 토큰은 로그에 남기 때문에 토큰을 첫 메시지로 보낸다(docs/api.md).
  // 서버가 4401로 닫으면 그때 쓴 토큰을 rejectedToken으로 넘기고, 아직 같은 토큰이면 갱신해 새 토큰을
  // 준다(c005648). 호출 위치: translation/api.ts의 liveToken(→ live/socket.ts).
  async liveToken(rejectedToken?: string): Promise<string> {
    const generation = this.generation
    if (!this.accessToken || this.accessToken === rejectedToken) await this.refresh()
    this.assertCurrent(generation)
    if (!this.accessToken) throw new SessionChangedError()
    return this.accessToken
  }

  // 인증이 필요한 API 요청. JSON·multipart·음성 파일 응답 모두 이 함수로 보낸다.
  // - /api/로 시작하지 않는 주소는 거부해 다른 곳으로 토큰이 나가지 않게 한다
  // - 메모리에 access 토큰이 없으면(새로 고침 직후) 먼저 갱신한다
  // - 401이면 한 번만 갱신 후 다시 보낸다. 방금 갱신한 토큰으로도 401이면 더 시도하지 않고 만료 로그아웃한다
  // - 실패 응답은 checked가 path에 맞는 안내 문구로 바꾼다
  // 호출 위치: translation/api.ts, history/api.ts, 아래 initialize·login의 /api/auth/me.
  /** 인증된 API 요청. 앞으로 생길 JSON·multipart·음성 응답 요청도 모두 이것을 쓴다. */
  async request(path: string, init: RequestInit = {}): Promise<Response> {
    if (!path.startsWith('/api/')) throw new Error('API paths must start with /api/')
    const generation = this.generation
    let refreshed = false
    if (!this.accessToken) {
      await this.refresh()
      refreshed = true
    }
    this.assertCurrent(generation)
    // 보낼 때마다 그 시점의 access 토큰으로 Authorization 머리글을 새로 만든다(갱신 뒤 다시 보낼 때 새 토큰이
    // 붙게). cache: 'no-store'로 브라우저 HTTP 캐시에 인증 응답이 남지 않게 한다.
    const send = () => {
      const headers = new Headers(init.headers)
      headers.set('Authorization', `Bearer ${this.accessToken}`)
      return fetch(path, { ...init, headers, cache: 'no-store' })
    }
    // 첫 전송에 쓴 토큰을 기억해, 401 뒤 다른 요청이 이미 토큰을 바꿨는지 알 수 있게 한다.
    const sentToken = this.accessToken
    let response = await send()
    this.assertCurrent(generation)
    if (response.status === 401 && !refreshed) {
      // 동시에 진행된 다른 요청이 거절된 토큰을 이미 바꿨을 수 있다.
      if (this.accessToken === sentToken) await this.refresh()
      this.assertCurrent(generation)
      response = await send()
      this.assertCurrent(generation)
    }
    // 갱신한 토큰으로도 401이면 서버가 이 사용자를 인정하지 않는 것(사용자 삭제 등)이므로 만료 로그아웃한다.
    if (response.status === 401) this.logout(true)
    return checked(response, path)
  }

  // 앱을 처음 열 때 세션을 복원한다(App.tsx의 첫 effect). 한 번 시작한 복원 약속을 기억해 StrictMode가
  // effect를 두 번 불러도 요청은 한 번이다.
  // - refresh 토큰이 없으면 서버에 묻지 않고 바로 anonymous
  // - 있으면 request('/api/auth/me')가 갱신 후 사용자 정보를 받아 authenticated
  // - 실패: 세션이 그 사이 바뀌었으면 무시, 토큰이 거절됐으면 만료 로그아웃, 서버·네트워크 오류면 토큰을 남긴
  //   채 만료 안내 없이 anonymous로 둔다. 서버가 돌아온 뒤 새로 고침하면 다시 복원된다(0db7b87)
  initialize = (): Promise<void> => {
    if (this.initializing) return this.initializing
    const generation = this.generation
    this.initializing = (async () => {
      try {
        if (!this.storage.getItem(REFRESH_KEY)) {
          this.publish({ status: 'anonymous', user: null, expired: false })
          return
        }
        const user: User = await (await this.request('/api/auth/me')).json()
        this.assertCurrent(generation)
        this.publish({ status: 'authenticated', user, expired: false })
      } catch (error) {
        if (generation !== this.generation) return
        if (authRejected(error)) this.logout(true)
        else this.publish({ status: 'anonymous', user: null, expired: false })
      }
    })()
    return this.initializing
  }

  // 회원가입. 인증이 필요 없어 request 대신 fetch를 바로 쓴다. 본문은 JSON(docs/api.md). 가입은 토큰을 주지
  // 않으므로 세션을 바꾸지 않는다.
  async register(email: string, password: string): Promise<User> {
    const response = await checked(await fetch('/api/auth/register', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    }))
    return response.json()
  }

  // 로그인. 먼저 기존 세션을 끝내(세대 증가) 이전 계정의 진행 중인 응답이 섞이지 않게 한다.
  // 서버의 로그인은 OAuth2 password 폼이라 JSON이 아니라 form-urlencoded로 보내고 이메일을 username 필드에
  // 넣는다(docs/api.md). 토큰을 저장한 뒤 /api/auth/me로 사용자를 확인하고 나서야 authenticated를 알린다.
  // 실패하면 반쯤 저장된 토큰을 지우고 오류를 화면으로 넘긴다.
  async login(email: string, password: string): Promise<void> {
    this.logout()
    const generation = this.generation
    try {
      const response = await checked(await fetch('/api/auth/login', {
        method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({ username: email, password }), cache: 'no-store',
      }))
      const tokens: Tokens = await response.json()
      this.assertCurrent(generation)
      this.saveTokens(tokens)
      const user: User = await (await this.request('/api/auth/me')).json()
      this.assertCurrent(generation)
      this.publish({ status: 'authenticated', user, expired: false })
    } catch (error) {
      if (generation === this.generation) this.logout()
      throw error
    }
  }
}

// 앱 전체가 함께 쓰는 클라이언트 하나. 토큰과 세션 상태가 한곳에만 있게 한다.
export const api = new ApiClient()

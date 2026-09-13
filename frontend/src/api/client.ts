export interface User {
  id: string
  email: string
  created_at: string
}

interface Tokens {
  access_token: string
  refresh_token: string
  token_type: string
  expires_in: number
}

export interface Session {
  status: 'loading' | 'anonymous' | 'authenticated'
  user: User | null
  expired: boolean
}

export const REFRESH_KEY = 'voice-translator.refresh'

export class ApiError extends Error {
  readonly status: number
  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

class SessionChangedError extends Error {
  constructor() { super('로그인 상태가 바뀌었습니다. 다시 시도해 주세요.') }
}

async function checked(response: Response, path = '/api/auth/login'): Promise<Response> {
  if (response.ok) return response
  // Only known status/detail pairs select local text. Never echo response inputs.
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
  const translationErrors: Record<string, string> = {
    '413:Audio file is larger than 10 MB': '음성 파일은 10MB 이하로 선택해 주세요.',
    '422:Audio could not be decoded': '음성 파일을 읽을 수 없습니다. 다시 녹음하거나 다른 파일을 선택해 주세요.',
    '422:Audio is longer than 30 seconds': '음성은 30초 이하로 녹음해 주세요.',
    '422:No speech was recognized': '말소리를 찾지 못했습니다. 마이크를 확인하고 다시 녹음해 주세요.',
    '503:Translation service is unavailable': '번역 서비스를 사용할 수 없습니다. 잠시 후 다시 시도해 주세요.',
    '404:Audio not found': '번역 음성이 없거나 삭제되었습니다. 다시 번역해 주세요.',
  }
  if (path.startsWith('/api/translate/') || path.startsWith('/api/audio/')) {
    const body: unknown = await response.json().catch(() => null)
    const detail = body && typeof body === 'object' && 'detail' in body ? body.detail : null
    const message = typeof detail === 'string' ? translationErrors[`${response.status}:${detail}`] : undefined
    if (message) throw new ApiError(response.status, message)
  }
  throw new ApiError(response.status, messages[response.status] ?? '요청을 처리하지 못했습니다. 잠시 후 다시 시도해 주세요.')
}

export function errorMessage(error: unknown): string {
  if (error instanceof ApiError || error instanceof SessionChangedError) return error.message
  return '서버에 연결할 수 없습니다. 연결을 확인하고 다시 시도해 주세요.'
}

export class ApiClient {
  private accessToken: string | null = null
  private generation = 0
  private refreshing: Promise<void> | null = null
  private initializing: Promise<void> | null = null
  private listeners = new Set<() => void>()
  private session: Session = { status: 'loading', user: null, expired: false }
  private readonly storage: Storage

  constructor(storage: Storage = window.sessionStorage) { this.storage = storage }

  getSnapshot = (): Session => this.session
  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener)
    return () => { this.listeners.delete(listener) }
  }

  private publish(session: Session) {
    this.session = session
    this.listeners.forEach((listener) => listener())
  }

  private assertCurrent(generation: number) {
    if (generation !== this.generation) throw new SessionChangedError()
  }

  private saveTokens(tokens: Tokens) {
    this.storage.setItem(REFRESH_KEY, tokens.refresh_token)
    this.accessToken = tokens.access_token
  }

  logout = (expired = false) => {
    this.generation += 1
    this.accessToken = null
    this.refreshing = null
    this.storage.removeItem(REFRESH_KEY)
    this.publish({ status: 'anonymous', user: null, expired })
  }

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
        if (generation === this.generation) this.logout(true)
        throw error
      }
    })()
    this.refreshing = pending
    // An older request must not clear a newer session's refresh.
    void pending.then(() => {
      if (this.refreshing === pending) this.refreshing = null
    }, () => {
      if (this.refreshing === pending) this.refreshing = null
    })
    return pending
  }

  /** Authenticated API requests, including future JSON, multipart, and audio responses. */
  async request(path: string, init: RequestInit = {}): Promise<Response> {
    if (!path.startsWith('/api/')) throw new Error('API paths must start with /api/')
    const generation = this.generation
    let refreshed = false
    if (!this.accessToken) {
      await this.refresh()
      refreshed = true
    }
    this.assertCurrent(generation)
    const send = () => {
      const headers = new Headers(init.headers)
      headers.set('Authorization', `Bearer ${this.accessToken}`)
      return fetch(path, { ...init, headers, cache: 'no-store' })
    }
    const sentToken = this.accessToken
    let response = await send()
    this.assertCurrent(generation)
    if (response.status === 401 && !refreshed) {
      // Another concurrent request may already have replaced the rejected token.
      if (this.accessToken === sentToken) await this.refresh()
      this.assertCurrent(generation)
      response = await send()
      this.assertCurrent(generation)
    }
    if (response.status === 401) this.logout(true)
    return checked(response, path)
  }

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
      } catch {
        if (generation === this.generation) this.logout(true)
      }
    })()
    return this.initializing
  }

  async register(email: string, password: string): Promise<User> {
    const response = await checked(await fetch('/api/auth/register', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    }))
    return response.json()
  }

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

export const api = new ApiClient()

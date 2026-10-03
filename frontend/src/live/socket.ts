import type { TranslationApi, TranslationResult } from '../translation/api'

export type LiveSocket = Pick<WebSocket, 'send' | 'close' | 'readyState' | 'bufferedAmount' | 'onopen' | 'onmessage' | 'onclose' | 'onerror'>
export type OpenSocket = (url: string) => LiveSocket
type Callbacks = { ready: (ready: boolean) => void; message: (message: unknown) => void; error: (text: string) => void }
export const liveUrl = (location: Pick<Location, 'protocol' | 'host'> = window.location) =>
  `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/api/translate/live`

// The server's per-utterance error details (docs/api.md), in words for someone who just spoke.
const utteranceErrors: Record<string, string> = {
  'Audio could not be decoded': '음성을 읽을 수 없습니다. 다시 말해 주세요.',
  'Audio is longer than 30 seconds': '음성은 30초 이하로 말해 주세요.',
  'No speech was recognized': '말소리를 찾지 못했습니다. 마이크를 확인해 주세요.',
  'Translation service is unavailable': '번역 서비스를 사용할 수 없습니다. 잠시 후 다시 말해 주세요.',
  'The translation could not be saved': '번역 결과를 저장하지 못했습니다. 다시 말해 주세요.',
}
export const utteranceError = (detail: unknown) =>
  (typeof detail === 'string' ? utteranceErrors[detail] : undefined) ?? '이 말을 번역하지 못했습니다. 다시 말해 주세요.'

export function isTranslationResult(value: unknown): value is TranslationResult {
  if (!value || typeof value !== 'object') return false
  const result = value as Partial<TranslationResult>
  return typeof result.id === 'string' && result.mode === 'speech' && typeof result.source_text === 'string'
    && typeof result.translated_text === 'string' && ['ko', 'en'].includes(result.source_lang ?? '')
    && ['ko', 'en'].includes(result.target_lang ?? '') && (result.audio_id === null || typeof result.audio_id === 'string')
}

export class LiveTransport {
  private socket?: LiveSocket
  private timeout?: ReturnType<typeof setTimeout>
  private retried = false
  private ready = false
  private ended = false
  private resolveReady?: () => void
  private rejectReady?: (error: Error) => void
  private readonly api: Pick<TranslationApi, 'liveToken' | 'expireLiveAuthentication'>
  private readonly fields: () => object
  private readonly callbacks: Callbacks
  private readonly open: OpenSocket
  private readonly name: string
  // `fields` go into every start message (direction, mode); read again when a new token reconnects.
  // `name` is the feature the connection error texts name.
  constructor(api: Pick<TranslationApi, 'liveToken' | 'expireLiveAuthentication'>, fields: () => object,
    callbacks: Callbacks, open: OpenSocket = (url) => new WebSocket(url), name = '동시통역') {
    this.api = api; this.fields = fields; this.callbacks = callbacks; this.open = open; this.name = name
  }

  start(signal: AbortSignal): Promise<void> {
    const ready = new Promise<void>((resolve, reject) => { this.resolveReady = resolve; this.rejectReady = reject })
    signal.addEventListener('abort', () => this.stop(), { once: true })
    if (signal.aborted) this.stop()
    else void this.connect()
    return ready
  }

  private async connect(rejectedToken?: string) {
    try {
      const token = await this.api.liveToken(rejectedToken)
      if (this.ended) return
      const socket = this.open(liveUrl())
      this.socket = socket
      const current = () => !this.ended && this.socket === socket
      this.timeout = setTimeout(() => this.fail(`${this.name} 연결 시간이 초과되었습니다. 다시 시작해 주세요.`), 12_000)
      socket.onopen = () => {
        if (!current()) return
        try { socket.send(JSON.stringify({ type: 'start', token, ...this.fields() })) }
        catch { this.fail(`${this.name} 서버에 연결할 수 없습니다. 다시 시작해 주세요.`) }
      }
      socket.onmessage = (event) => {
        if (!current()) return
        try {
          const message: unknown = JSON.parse(event.data)
          if (typeof message !== 'object' || message === null || !('type' in message)) throw new Error()
          if (message.type === 'ready') {
            if (this.ready) return
            clearTimeout(this.timeout)
            this.ready = true
            // The server accepted this token, so a later 4401 is the next expiry and gets its own retry.
            // The one-retry budget only stops a loop when a freshly refreshed token is rejected before ready.
            this.retried = false
            this.callbacks.ready(true)
            this.resolveReady?.()
          } else if (this.ready) this.callbacks.message(message)
          else throw new Error()
        } catch { this.fail(`${this.name} 응답을 읽을 수 없습니다. 다시 시작해 주세요.`) }
      }
      socket.onerror = () => { if (current()) this.fail(`${this.name} 서버에 연결할 수 없습니다. 연결을 확인해 주세요.`) }
      socket.onclose = (event) => {
        if (!current()) return
        clearTimeout(this.timeout)
        this.ready = false
        this.detach()
        this.callbacks.ready(false)
        if (event.code === 4401 && !this.retried) {
          this.retried = true
          void this.connect(token)
        } else {
          if (event.code === 4401) this.api.expireLiveAuthentication()
          this.fail(event.code === 4401 ? '로그인이 만료되었습니다. 다시 로그인해 주세요.'
            : event.code === 4503 ? `${this.name} 서비스를 사용할 수 없습니다. 잠시 후 다시 시작해 주세요.`
            : event.code === 4422 ? `${this.name} 연결 요청이 올바르지 않습니다. 다시 시작해 주세요.`
            : `${this.name} 연결이 끊겼습니다. 연결을 확인하고 다시 시작해 주세요.`)
        }
      }
    } catch {
      if (!this.ended) this.fail(`${this.name} 인증을 확인하지 못했습니다. 로그인 상태와 연결을 확인해 주세요.`)
    }
  }

  send(message: object | ArrayBuffer) {
    if (!this.ready || !this.socket || this.socket.readyState !== 1) return
    // Bound queued audio if the network stops accepting data (roughly one minute of PCM).
    if (this.socket.bufferedAmount > 2_000_000) {
      this.fail('음성 전송이 지연되어 멈췄습니다. 연결을 확인하고 다시 시작해 주세요.')
      return
    }
    try { this.socket.send(message instanceof ArrayBuffer ? message : JSON.stringify(message)) }
    catch { this.fail('음성을 전송하지 못했습니다. 다시 시작해 주세요.') }
  }

  private detach() {
    const socket = this.socket
    this.socket = undefined
    if (socket) socket.onopen = socket.onmessage = socket.onclose = socket.onerror = null
    return socket
  }
  private fail(text: string) {
    if (this.ended) return
    this.stop()
    this.callbacks.error(text)
  }
  stop() {
    this.ended = true; this.ready = false
    clearTimeout(this.timeout)
    this.rejectReady?.(new Error('Live connection stopped'))
    this.detach()?.close()
  }
}

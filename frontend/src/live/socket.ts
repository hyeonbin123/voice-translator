import type { Direction, TranslationApi } from '../translation/api'

export type LiveSocket = Pick<WebSocket, 'send' | 'close' | 'readyState' | 'bufferedAmount' | 'onopen' | 'onmessage' | 'onclose' | 'onerror'>
export type OpenSocket = (url: string) => LiveSocket
type Callbacks = { ready: (ready: boolean) => void; message: (message: unknown) => void; error: (text: string) => void }
export const liveUrl = (location: Pick<Location, 'protocol' | 'host'> = window.location) =>
  `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/api/translate/live`

export class LiveTransport {
  private socket?: LiveSocket
  private timeout?: ReturnType<typeof setTimeout>
  private retried = false
  private ready = false
  private ended = false
  private resolveReady?: () => void
  private rejectReady?: (error: Error) => void
  private readonly api: Pick<TranslationApi, 'liveToken' | 'expireLiveAuthentication'>
  private readonly direction: Direction
  private readonly callbacks: Callbacks
  private readonly open: OpenSocket
  constructor(api: Pick<TranslationApi, 'liveToken' | 'expireLiveAuthentication'>, direction: Direction,
    callbacks: Callbacks, open: OpenSocket = (url) => new WebSocket(url)) {
    this.api = api; this.direction = direction; this.callbacks = callbacks; this.open = open
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
      this.timeout = setTimeout(() => this.fail('동시통역 연결 시간이 초과되었습니다. 다시 시작해 주세요.'), 12_000)
      socket.onopen = () => {
        if (!current()) return
        try { socket.send(JSON.stringify({ type: 'start', token, ...this.direction })) }
        catch { this.fail('동시통역 서버에 연결할 수 없습니다. 다시 시작해 주세요.') }
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
            this.callbacks.ready(true)
            this.resolveReady?.()
          } else if (this.ready) this.callbacks.message(message)
          else throw new Error()
        } catch { this.fail('동시통역 응답을 읽을 수 없습니다. 다시 시작해 주세요.') }
      }
      socket.onerror = () => { if (current()) this.fail('동시통역 서버에 연결할 수 없습니다. 연결을 확인해 주세요.') }
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
            : event.code === 4503 ? '동시통역 서비스를 사용할 수 없습니다. 잠시 후 다시 시작해 주세요.'
            : event.code === 4422 ? '동시통역 연결 요청이 올바르지 않습니다. 다시 시작해 주세요.'
            : '동시통역 연결이 끊겼습니다. 연결을 확인하고 다시 시작해 주세요.')
        }
      }
    } catch {
      if (!this.ended) this.fail('동시통역 인증을 확인하지 못했습니다. 로그인 상태와 연결을 확인해 주세요.')
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

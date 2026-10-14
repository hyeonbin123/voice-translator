// [파일 역할] WS /api/translate/live 웹소켓 연결 하나를 맡는 전송 층(LiveTransport)과, 세 기능(동시통역,
// 로그인한 대화 모드, 두 사람 대화)이 함께 쓰는 마디 오류 문구·최종 결과 검사.
// [흐름] 토큰 받기 → 소켓 열기 → 열리면 첫 메시지 start(토큰 + 방향·모드)를 보냄 → 서버가 ready로 답하면 준비
// 완료 → 이후 메시지는 콜백으로 넘기고 send로 JSON·소리를 보냄 → 닫히면 닫힘 코드에 맞는 안내를 띄우고 멈춤.
// [설계 이유] 브라우저 WebSocket은 인증 머리글을 붙일 수 없고 주소의 토큰은 로그에 남으므로 토큰은 연결 뒤 첫
// 메시지로만 보낸다(docs/api.md). 서버는 access 토큰이 만료되면 다음 메시지에서 4401로 닫으므로, 그때 새
// 토큰으로 한 번 다시 연결한다. 이 재시도 기회는 서버가 ready로 받아 줄 때마다 다시 채워 30분마다 오는
// 만료에도 계속 이어지고, 새 토큰마저 ready 전에 거절되면 그때 로그아웃해 무한 재연결을 막는다(c90d1ee).
// [연결] live/session.ts(동시통역)와 conversation/speculative.ts(대화 모드·두 사람 대화의 미리 처리)가
// LiveTransport를 만든다. 테스트는 open 인자로 test/fakeSocket.ts의 가짜 소켓을 넣는다.

import type { TranslationApi, TranslationResult } from '../translation/api'

// LiveSocket: 이 파일이 실제로 쓰는 WebSocket 기능만 뽑은 타입. 테스트의 가짜 소켓이 이 모양만 갖추면 된다.
// OpenSocket: 주소를 받아 소켓을 여는 함수(기본은 new WebSocket).
export type LiveSocket = Pick<WebSocket, 'send' | 'close' | 'readyState' | 'bufferedAmount' | 'onopen' | 'onmessage' | 'onclose' | 'onerror'>
export type OpenSocket = (url: string) => LiveSocket
type Callbacks = { ready: (ready: boolean) => void; message: (message: unknown) => void; error: (text: string) => void }
// 웹소켓 주소. 페이지가 https면 wss, 아니면 ws를 쓰고 같은 호스트의 /api/translate/live로 붙는다. 배포에서는
// nginx가, 개발에서는 Vite 프록시가 이 경로를 API 서버로 넘긴다.
export const liveUrl = (location: Pick<Location, 'protocol' | 'host'> = window.location) =>
  `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/api/translate/live`

// 서버의 마디별 오류 detail(docs/api.md)을 방금 말한 사람에게 맞는 말로 옮긴 표.
const utteranceErrors: Record<string, string> = {
  'Audio could not be decoded': '음성을 읽을 수 없습니다. 다시 말해 주세요.',
  'Audio is longer than 30 seconds': '음성은 30초 이하로 말해 주세요.',
  'No speech was recognized': '말소리를 찾지 못했습니다. 마이크를 확인해 주세요.',
  'Translation service is unavailable': '번역 서비스를 사용할 수 없습니다. 잠시 후 다시 말해 주세요.',
  'The translation could not be saved': '번역 결과를 저장하지 못했습니다. 다시 말해 주세요.',
}
// 서버의 마디 오류(error 메시지의 detail)를 방금 말한 사람에게 맞는 한국어 안내로 바꾼다. 표에 없는 detail은
// 일반 문구로 바꿔 서버 문자열이 화면에 그대로 나오지 않게 한다. 쓰는 곳: live/session.ts,
// conversation/speculative.ts.
export const utteranceError = (detail: unknown) =>
  (typeof detail === 'string' ? utteranceErrors[detail] : undefined) ?? '이 말을 번역하지 못했습니다. 다시 말해 주세요.'

// final 메시지의 result가 화면이 믿고 쓸 수 있는 음성 번역 결과인지 확인하는 타입 가드. 웹소켓 메시지는 타입
// 검사를 거치지 않은 JSON이라, 화면이 쓰는 필드(id, mode 'speech', 원문·번역문, 두 언어, audio_id)를 직접
// 확인한다.
export function isTranslationResult(value: unknown): value is TranslationResult {
  if (!value || typeof value !== 'object') return false
  const result = value as Partial<TranslationResult>
  return typeof result.id === 'string' && result.mode === 'speech' && typeof result.source_text === 'string'
    && typeof result.translated_text === 'string' && ['ko', 'en'].includes(result.source_lang ?? '')
    && ['ko', 'en'].includes(result.target_lang ?? '') && (result.audio_id === null || typeof result.audio_id === 'string')
}

// 웹소켓 연결 하나의 수명을 관리한다.
// - retried: 4401 뒤 재연결을 이미 한 번 했는지. ready를 받으면 false로 되돌린다
// - ready: 서버가 start를 받아 준 뒤인지. 그 전에는 send가 아무것도 보내지 않는다
// - ended: stop 뒤에는 늦게 도착한 이벤트·토큰을 모두 무시한다
// - resolveReady·rejectReady: start가 돌려준 약속을 준비 완료나 중단 때 끝낸다
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
  // `fields`는 모든 start 메시지에 들어가고(방향, 모드), 새 토큰으로 다시 연결할 때 다시 읽는다.
  // `name`은 연결 오류 문구에 들어가는 기능 이름이다.
  constructor(api: Pick<TranslationApi, 'liveToken' | 'expireLiveAuthentication'>, fields: () => object,
    callbacks: Callbacks, open: OpenSocket = (url) => new WebSocket(url), name = '동시통역') {
    this.api = api; this.fields = fields; this.callbacks = callbacks; this.open = open; this.name = name
  }

  // 연결을 시작하고, 서버가 ready로 답하면 끝나는 약속을 돌려준다. signal이 취소되면(사용자가 멈추거나 화면을
  // 떠남) 연결을 닫는다. 이미 취소된 signal이면 연결하지 않는다.
  start(signal: AbortSignal): Promise<void> {
    const ready = new Promise<void>((resolve, reject) => { this.resolveReady = resolve; this.rejectReady = reject })
    signal.addEventListener('abort', () => this.stop(), { once: true })
    if (signal.aborted) this.stop()
    else void this.connect()
    return ready
  }

  // 토큰을 받아 소켓을 연다. rejectedToken은 4401로 거절된 토큰으로, liveToken이 같은 토큰이면 갱신해 새
  // 토큰을 준다.
  // current(): 이 소켓이 아직 지금 연결인지. 다시 연결하거나 멈춘 뒤 옛 소켓의 이벤트가 늦게 와도 무시하게
  // 한다.
  // 12초 시간 제한: 서버는 연결 뒤 10초 안에 start가 와야 받으므로(docs/api.md), 그보다 조금 길게 기다려도
  // ready가 없으면 실패로 본다.
  private async connect(rejectedToken?: string) {
    try {
      const token = await this.api.liveToken(rejectedToken)
      if (this.ended) return
      const socket = this.open(liveUrl())
      this.socket = socket
      const current = () => !this.ended && this.socket === socket
      this.timeout = setTimeout(() => this.fail(`${this.name} 연결 시간이 초과되었습니다. 다시 시작해 주세요.`), 12_000)
      // 열리자마자 첫 메시지 start를 보낸다. fields()를 매번 다시 읽는 이유는 재연결 때 지금의 방향·모드(두
      // 사람 대화는 직전 마디 언어와 id)를 실어야 하기 때문이다.
      socket.onopen = () => {
        if (!current()) return
        try { socket.send(JSON.stringify({ type: 'start', token, ...this.fields() })) }
        catch { this.fail(`${this.name} 서버에 연결할 수 없습니다. 다시 시작해 주세요.`) }
      }
      // 서버 메시지 처리. type이 있는 JSON 객체가 아니면 연결을 끊는다. ready는 한 번만 받아들이고 그 뒤
      // 메시지는 콜백(message)으로 넘긴다. ready 전에 다른 메시지가 오면 계약 위반으로 본다.
      socket.onmessage = (event) => {
        if (!current()) return
        try {
          const message: unknown = JSON.parse(event.data)
          if (typeof message !== 'object' || message === null || !('type' in message)) throw new Error()
          if (message.type === 'ready') {
            if (this.ready) return
            clearTimeout(this.timeout)
            this.ready = true
            // 서버가 이 토큰을 받아들였으므로, 나중의 4401은 다음 만료이고 다시 한 번 재시도할 수 있다. 재시도
            // 한 번 제한은 방금 갱신한 토큰이 ready 전에 거절될 때 생기는 반복만 막는다.
            this.retried = false
            this.callbacks.ready(true)
            this.resolveReady?.()
          } else if (this.ready) this.callbacks.message(message)
          else throw new Error()
        } catch { this.fail(`${this.name} 응답을 읽을 수 없습니다. 다시 시작해 주세요.`) }
      }
      // 네트워크 오류. 실제 이유는 브라우저가 알려 주지 않으므로 연결 확인 안내만 띄운다.
      socket.onerror = () => { if (current()) this.fail(`${this.name} 서버에 연결할 수 없습니다. 연결을 확인해 주세요.`) }
      // 닫힘 처리. 먼저 준비 상태를 끄고 화면에 알린다.
      // - 4401(토큰 없음·틀림·만료)이고 아직 재시도하지 않았으면 그 토큰을 거절된 것으로 넘겨 새 토큰으로
      //   다시 연결한다
      // - 재시도한 뒤에도 4401이면 로그인이 정말 끝난 것이므로 만료 로그아웃(expireLiveAuthentication)하고
      //   안내한다
      // - 4503(서버에 모델 없음), 4422(요청 형식 오류), 그 밖의 닫힘은 각각의 안내를 띄우고 멈춘다(닫힘
      //   코드는 docs/api.md)
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

  // 메시지 보내기. JSON 객체는 문자열로, 소리(ArrayBuffer)는 바이너리 메시지로 보낸다. 준비 전이거나 소켓이
  // 열려 있지 않으면 조용히 버린다. 보내기 실패는 연결 오류로 처리한다.
  send(message: object | ArrayBuffer) {
    if (!this.ready || !this.socket || this.socket.readyState !== 1) return
    // 2,000,000바이트는 16kHz·16비트 모노 PCM(초당 32,000바이트)으로 약 1분 분량이다. 네트워크가 멈춰 보내지
    // 못한 소리가 이만큼 쌓이면 메모리가 끝없이 늘지 않게 멈춘다.
    if (this.socket.bufferedAmount > 2_000_000) {
      this.fail('음성 전송이 지연되어 멈췄습니다. 연결을 확인하고 다시 시작해 주세요.')
      return
    }
    try { this.socket.send(message instanceof ArrayBuffer ? message : JSON.stringify(message)) }
    catch { this.fail('음성을 전송하지 못했습니다. 다시 시작해 주세요.') }
  }

  // 지금 소켓의 이벤트 처리기를 모두 떼고 참조를 지운 뒤 그 소켓을 돌려준다. 닫는 중인 옛 소켓의 이벤트가 새
  // 상태를 건드리지 않게 한다.
  private detach() {
    const socket = this.socket
    this.socket = undefined
    if (socket) socket.onopen = socket.onmessage = socket.onclose = socket.onerror = null
    return socket
  }
  // 연결을 멈추고 오류 안내를 한 번만 띄운다(이미 멈췄으면 아무것도 하지 않음).
  private fail(text: string) {
    if (this.ended) return
    this.stop()
    this.callbacks.error(text)
  }
  // 연결을 끝낸다: 상태를 끝으로 바꾸고, 시간 제한을 지우고, 기다리던 start 약속을 실패로 끝낸 뒤 소켓을
  // 닫는다. 여러 번 불러도 안전하다.
  stop() {
    this.ended = true; this.ready = false
    clearTimeout(this.timeout)
    this.rejectReady?.(new Error('Live connection stopped'))
    this.detach()?.close()
  }
}

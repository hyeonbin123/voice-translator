// [파일 역할] 로그인한 대화 모드와 두 사람 대화가 쓰는 '미리 처리' 채널. 말 끝 판정기(스트리밍 모드)의
// 이벤트를 웹소켓 메시지로 바꿔 보내고, 끝난 마디마다 서버의 최종 결과를 약속(Promise)으로 돌려준다.
// [왜 필요한가] 예전에는 1초 쉼이 확인된 뒤에야 녹음을 HTTP로 올려, 말을 멈춘 뒤 번역 음성까지 약 2초가
// 걸렸다. 마디를 말하는 동안 소리를 흘려보내면 서버가 192ms 쉼(pause)에서 인식·번역·합성을 미리 시작하고 1초
// 판정(end)에서 확정만 하므로, 측정에서 대화 모드 test 중앙값이 한국어 원문 1.86 → 1.24초, 영어 원문 2.25 →
// 1.43초로 줄었고 160문장 모두 결과가 업로드 방식과 같았다(README, docs/experiments.md 11절).
// [메시지 대응] start → utterance(새 마디 번호), audio → 16비트 PCM 바이너리, pause·resume → 그대로, discard
// → cancel, end → end 후 결과 기다림. 서버의 final은 결과로, error는 그 마디의 오류로 바꾼다(docs/api.md WS
// /api/translate/live).
// [연결] conversation/session.ts와 dialog/session.ts가 만든다. 연결 관리(토큰, 재연결, 닫힘 코드)는
// live/socket.ts의 LiveTransport에 맡긴다.

import { ApiError } from '../api/client'
import type { TranslationApi, TranslationResult } from '../translation/api'
import { isTranslationResult, LiveTransport, utteranceError, type OpenSocket } from '../live/socket'
import type { LiveEndpointEvent } from './silero'
import { encodePcm16 } from './pcm'

/** 끝났지만 결과가 오지 않을 마디: 연결이 멈췄거나 새 토큰으로 다시 연결됐다. */
export class CanceledUtterance extends Error {
  constructor() { super('The utterance was canceled'); this.name = 'CanceledUtterance' }
}
// end로 확정한 마디: 브라우저가 붙인 마디 번호와, final이 오면 풀리는 결과 약속.
export interface Commit<T> { utterance: number; result: Promise<T> }
// 토큰 만료로 다시 연결하는 동안 세션이 띄우는 안내. 재연결하면 서버가 들고 있던 마디는 사라지므로 다시 말해
// 달라고 한다.
export const RECONNECTING = '인증을 새로 확인하고 연결하는 중입니다. 듣는 중 표시 뒤 다시 말해 주세요.'
type Callbacks = { ready: (ready: boolean) => void; error: (text: string) => void }
type Waiting<T> = { resolve: (result: T) => void; reject: (error: Error) => void }

/**
 * 자막 없이 동시통역 웹소켓을 쓰는 대화 모드와 두 사람 대화(T77, docs/api.md): 마디를 말하는 동안 소리를
 * 흘려보내고, 서버는 192 ms 쉼에서 번역을 미리 준비했다가 말이 이어지면 버리며, end가 오면 확정한다. 결과는
 * 같은 클립에 대한 HTTP API의 결과와 같고, 더 빨리 나온다.
 */
// T: 결과 타입(대화 모드는 음성 번역 결과, 두 사람 대화는 언어 판별 필드가 붙은 결과). isResult로 final의
// result가 그 타입인지 확인한다.
// - current: 지금 보내고 있는 마디 번호(마디 밖이면 null)
// - nextId: 다음 마디 번호. 채널(대화 한 번)마다 1부터 늘어나고, 토큰 만료로 다시 연결해도 이어진다
// - waiting: end를 보내고 결과를 기다리는 마디들
// - ready: 서버가 start를 받아 줘 메시지를 보낼 수 있는지. 세션은 준비되지 않은 동안 마이크를 쉬게 한다
export class SpeculativeChannel<T extends TranslationResult = TranslationResult> {
  private readonly transport: LiveTransport
  private readonly isResult: (value: unknown) => value is T
  private current: number | null = null
  private nextId = 1
  private waiting = new Map<number, Waiting<T>>()
  ready = false

  // fields: 연결할 때마다 start에 넣을 값(방향과 mode 'conversation' 또는 'dialog'). name: 연결 오류 안내에
  // 들어갈 기능 이름. socket: 테스트용 가짜 소켓을 여는 함수.
  // 연결이 준비를 잃거나(재연결) 오류로 끝나면 기다리던 마디를 모두 CanceledUtterance로 끝낸다. 서버 쪽에서는
  // 그 마디들이 이미 버려졌기 때문이다.
  constructor(api: Pick<TranslationApi, 'liveToken' | 'expireLiveAuthentication'>, fields: () => object,
    callbacks: Callbacks, name: string, isResult: (value: unknown) => value is T = isTranslationResult as (value: unknown) => value is T,
    socket?: OpenSocket) {
    this.isResult = isResult
    this.transport = new LiveTransport(api, fields, {
      ready: (ready) => {
        this.ready = ready
        // 새 토큰은 새 연결이라는 뜻이다: 옛 연결이 들고 있던 것은 서버에서 사라졌다.
        if (!ready) { this.current = null; this.cancelWaiting() }
        callbacks.ready(ready)
      },
      message: (message) => this.receive(message),
      error: (text) => { this.ready = false; this.cancelWaiting(); callbacks.error(text) },
    }, socket, name)
  }

  // 연결 시작. 서버가 ready로 답하면 풀린다. 세션은 마이크·판정기 준비와 함께 이것을 기다린다.
  start(signal: AbortSignal) { return this.transport.start(signal) }

  /** 판정기 이벤트 하나를 넘긴다. end면 마디를 확정하고, 앞으로 올 결과를 돌려준다. */
  event(event: LiveEndpointEvent): Commit<T> | undefined {
    // start: 새 마디 번호를 정해 utterance를 보낸다. 서버는 그때 진행 중인 다른 마디가 있으면
    // 버린다(docs/api.md).
    if (event.type === 'start') {
      this.current = this.nextId++
      this.transport.send({ type: 'utterance', id: this.current })
      return undefined
    }
    // 마디 밖의 이벤트(재생 시작으로 취소한 마디의 나머지 등)는 보내지 않는다. end는 current를 비우고, 결과를
    // 기다릴 약속을 등록한 뒤 end를 보낸다.
    if (this.current === null) return undefined
    const id = this.current
    if (event.type === 'audio') this.transport.send(encodePcm16(event.samples))
    else if (event.type === 'pause' || event.type === 'resume') this.transport.send({ type: event.type, id })
    else if (event.type === 'discard') this.interrupt()
    else if (event.type === 'end') {
      this.current = null
      const result = new Promise<T>((resolve, reject) => { this.waiting.set(id, { resolve, reject }) })
      result.catch(() => undefined) // 세션이 차례가 오면 기다린다. 그 전에 거절돼도 미처리 거절이 되지 않게 한다
      this.transport.send({ type: 'end', id })
      return { utterance: id, result }
    }
    return undefined
  }

  /** 진행 중인 말을 버린다(재생 시작, 너무 짧음): 서버도 미리 준비한 것을 버린다. */
  // 진행 중인 마디가 있으면 cancel을 보내 서버가 미리 하던 처리와 받은 소리를 버리게 한다.
  interrupt() {
    if (this.current === null) return
    this.transport.send({ type: 'cancel', id: this.current })
    this.current = null
  }

  // 그 밖의 메시지를 그대로 보낸다. 두 사람 대화가 방향을 바꾼 마디를 알리는 previous 메시지에
  // 쓴다(dialog/session.ts).
  send(message: object) { this.transport.send(message) }

  // 멈춤: 진행 중인 마디를 취소하고, 기다리던 결과를 모두 취소로 끝낸 뒤 연결을 닫는다.
  stop() {
    this.interrupt()
    this.cancelWaiting()
    this.transport.stop()
  }

  // 기다리던 마디를 모두 CanceledUtterance로 끝낸다. 목록을 먼저 비운 뒤 거절해, 거절 처리 중에 목록이
  // 바뀌어도 안전하다.
  private cancelWaiting() {
    const waiting = [...this.waiting.values()]
    this.waiting.clear()
    for (const entry of waiting) entry.reject(new CanceledUtterance())
  }

  // 서버 메시지 처리. 기다리는 마디의 번호가 아니면 무시한다. final이고 결과 모양이 맞으면 그 약속을 풀고,
  // error면 화면용 안내 문구를 담은 ApiError로 거절한다. 모델 없음(503)과 그 밖의 마디 오류(422)를 상태
  // 코드로 나눠 HTTP 경로와 같은 오류 처리를 탄다.
  private receive(value: unknown) {
    if (!value || typeof value !== 'object' || !('id' in value) || !('type' in value)) return
    const entry = typeof value.id === 'number' ? this.waiting.get(value.id) : undefined
    if (!entry) return
    if (value.type === 'final' && 'result' in value && this.isResult(value.result)) {
      this.waiting.delete(value.id as number)
      entry.resolve(value.result)
    } else if (value.type === 'error') {
      this.waiting.delete(value.id as number)
      const detail = 'detail' in value ? value.detail : undefined
      entry.reject(new ApiError(detail === 'Translation service is unavailable' ? 503 : 422, utteranceError(detail)))
    }
  }
}

import { ApiError } from '../api/client'
import type { TranslationApi, TranslationResult } from '../translation/api'
import { isTranslationResult, LiveTransport, utteranceError, type OpenSocket } from '../live/socket'
import type { LiveEndpointEvent } from './silero'
import { encodePcm16 } from './pcm'

/** An ended utterance whose result will not come: the connection stopped or reconnected with a new token. */
export class CanceledUtterance extends Error {
  constructor() { super('The utterance was canceled'); this.name = 'CanceledUtterance' }
}
export interface Commit<T> { utterance: number; result: Promise<T> }
export const RECONNECTING = '인증을 새로 확인하고 연결하는 중입니다. 듣는 중 표시 뒤 다시 말해 주세요.'
type Callbacks = { ready: (ready: boolean) => void; error: (text: string) => void }
type Waiting<T> = { resolve: (result: T) => void; reject: (error: Error) => void }

/**
 * Conversation and dialog modes over the live WebSocket without subtitles (T77, docs/api.md): the
 * utterance's audio streams while it is spoken, the server prepares the translation at a 192 ms pause and
 * drops it if speech resumes, and an end commits it. The result is the HTTP API's for the same clip, sooner.
 */
export class SpeculativeChannel<T extends TranslationResult = TranslationResult> {
  private readonly transport: LiveTransport
  private readonly isResult: (value: unknown) => value is T
  private current: number | null = null
  private nextId = 1
  private waiting = new Map<number, Waiting<T>>()
  ready = false

  constructor(api: Pick<TranslationApi, 'liveToken' | 'expireLiveAuthentication'>, fields: () => object,
    callbacks: Callbacks, name: string, isResult: (value: unknown) => value is T = isTranslationResult as (value: unknown) => value is T,
    socket?: OpenSocket) {
    this.isResult = isResult
    this.transport = new LiveTransport(api, fields, {
      ready: (ready) => {
        this.ready = ready
        // A new token means a new connection: what the old one held is gone on the server.
        if (!ready) { this.current = null; this.cancelWaiting() }
        callbacks.ready(ready)
      },
      message: (message) => this.receive(message),
      error: (text) => { this.ready = false; this.cancelWaiting(); callbacks.error(text) },
    }, socket, name)
  }

  start(signal: AbortSignal) { return this.transport.start(signal) }

  /** Passes one endpointer event on; an end commits the utterance and returns its result to come. */
  event(event: LiveEndpointEvent): Commit<T> | undefined {
    if (event.type === 'start') {
      this.current = this.nextId++
      this.transport.send({ type: 'utterance', id: this.current })
      return undefined
    }
    if (this.current === null) return undefined
    const id = this.current
    if (event.type === 'audio') this.transport.send(encodePcm16(event.samples))
    else if (event.type === 'pause' || event.type === 'resume') this.transport.send({ type: event.type, id })
    else if (event.type === 'discard') this.interrupt()
    else if (event.type === 'end') {
      this.current = null
      const result = new Promise<T>((resolve, reject) => { this.waiting.set(id, { resolve, reject }) })
      result.catch(() => undefined) // the session awaits it in turn; a rejection before then is not unhandled
      this.transport.send({ type: 'end', id })
      return { utterance: id, result }
    }
    return undefined
  }

  /** Speech in progress is dropped (playback started, too short): the server drops what it prepared. */
  interrupt() {
    if (this.current === null) return
    this.transport.send({ type: 'cancel', id: this.current })
    this.current = null
  }

  send(message: object) { this.transport.send(message) }

  stop() {
    this.interrupt()
    this.cancelWaiting()
    this.transport.stop()
  }

  private cancelWaiting() {
    const waiting = [...this.waiting.values()]
    this.waiting.clear()
    for (const entry of waiting) entry.reject(new CanceledUtterance())
  }

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

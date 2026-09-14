import { afterEach, expect, it, vi } from 'vitest'
import { TranslationApi, type TranslationResult } from '../translation/api'
import type { OpenMicrophone } from '../conversation/microphone'
import type { PlayTranslation } from '../conversation/player'
import type { CreateEndpointer, LiveEndpointEvent } from '../conversation/silero'
import { LiveSession } from './session'
import { liveUrl, type LiveSocket } from './socket'

const direction = { source_lang: 'ko', target_lang: 'en' } as const
const result: TranslationResult = { id: 'saved-1', mode: 'speech', ...direction, source_text: '안녕하세요',
  translated_text: 'Hello', audio_id: null, tts_error: null, stt_model: 'stt', mt_model: 'mt', tts_model: null,
  stt_ms: 1, mt_ms: 1, tts_ms: null, created_at: '' }
function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => { resolve = done })
  return { promise, resolve }
}
class FakeSocket implements LiveSocket {
  readyState: WebSocket['readyState'] = 0; bufferedAmount = 0
  onopen: WebSocket['onopen'] = null
  onmessage: WebSocket['onmessage'] = null
  onclose: WebSocket['onclose'] = null
  onerror: WebSocket['onerror'] = null
  send = vi.fn<(data: string | ArrayBufferLike | Blob | ArrayBufferView) => void>()
  close = vi.fn(() => { this.readyState = 3 })
  opened() { this.readyState = 1; this.onopen?.call(this as unknown as WebSocket, new Event('open')) }
  message(data: unknown) { this.onmessage?.call(this as unknown as WebSocket, new MessageEvent('message', { data: JSON.stringify(data) })) }
  closed(code: number) { this.readyState = 3; this.onclose?.call(this as unknown as WebSocket, new CloseEvent('close', { code })) }
  messages() { return this.send.mock.calls.map(([data]) => typeof data === 'string' ? JSON.parse(data) : 'audio') }
}
const sessions: LiveSession[] = []
afterEach(() => { sessions.forEach((session) => session.stop()); sessions.length = 0; vi.useRealTimers() })

function setup(create?: CreateEndpointer) {
  let capture!: (samples: Float32Array) => void
  let lost!: () => void
  const microphone = { stop: vi.fn(), setPaused: vi.fn() }
  const open = vi.fn<OpenMicrophone>(async (handler, _signal, disconnected) => { capture = handler; lost = disconnected; return microphone })
  let events: LiveEndpointEvent[] = []
  const detector = { interrupt: vi.fn(() => true), push: vi.fn(async () => events) }
  const sockets: FakeSocket[] = []
  const urls: string[] = []
  const liveToken = vi.fn(async (rejected?: string): Promise<string> => rejected ? 'fresh' : 'access')
  const logout = vi.fn()
  const api = new TranslationApi({ request: vi.fn(), liveToken, logout })
  const played: { begin: () => void; signal: AbortSignal; task: ReturnType<typeof deferred<void>> }[] = []
  const play = vi.fn<PlayTranslation>((_result, signal, begin) => {
    const task = deferred<void>(); played.push({ signal, begin, task }); return task.promise
  })
  const session = new LiveSession(api, open, play, create ?? (async () => detector), (url) => {
    const socket = new FakeSocket(); sockets.push(socket); urls.push(url); return socket
  })
  sessions.push(session)
  const start = async () => {
    const starting = session.start(direction)
    await vi.waitFor(() => expect(sockets).toHaveLength(1))
    sockets[0].opened(); sockets[0].message({ type: 'ready' }); await starting
  }
  const emit = async (...next: LiveEndpointEvent[]) => {
    events = next; capture(new Float32Array(512)); await Promise.resolve(); await Promise.resolve()
  }
  return { session, sockets, urls, microphone, detector, liveToken, logout, played, play, start, emit,
    capture: (samples = new Float32Array(512)) => capture(samples), lost: () => lost() }
}
const onset: LiveEndpointEvent = { type: 'start', startSample: 0 }
const audio: LiveEndpointEvent = { type: 'audio', samples: new Float32Array([-1, -.5, 0, .5, 1]) }
const end: LiveEndpointEvent = { type: 'end', samples: new Float32Array(), startSample: 0, endSample: 0, detectedAtSample: 0, forced: false }

it('uses the same origin, sends the token only in start, and gates capture on ready', async () => {
  const s = setup(); const starting = s.session.start(direction)
  await vi.waitFor(() => expect(s.sockets).toHaveLength(1))
  s.capture(); expect(s.detector.push).not.toHaveBeenCalled()
  s.sockets[0].opened()
  expect(s.sockets[0].messages()).toEqual([{ type: 'start', token: 'access', ...direction }])
  expect(s.urls[0]).toBe(liveUrl())
  expect(s.urls[0]).not.toContain('access')
  expect(liveUrl({ protocol: 'https:', host: 'example.test:8443' })).toBe('wss://example.test:8443/api/translate/live')
  s.sockets[0].message({ type: 'ready' }); await starting
  expect(s.microphone.setPaused).toHaveBeenLastCalledWith(false)
})

it.each([false, true])('sends utterance/audio/pause/%s resume/pause/end in order with identical PCM16', async (resume) => {
  const s = setup(); await s.start()
  await s.emit(onset, audio, { type: 'pause' }, ...(resume ? [{ type: 'resume' }, audio, { type: 'pause' }] as LiveEndpointEvent[] : []), end)
  const expected = [{ type: 'utterance', id: 1 }, 'audio', { type: 'pause', id: 1 },
    ...(resume ? [{ type: 'resume', id: 1 }, 'audio', { type: 'pause', id: 1 }] : []), { type: 'end', id: 1 }]
  expect(s.sockets[0].messages().slice(1)).toEqual(expected)
  expect([...new Int16Array(s.sockets[0].send.mock.calls[2][0] as ArrayBuffer)]).toEqual([-32768, -16384, 0, 16384, 32767])
  expect(s.session.getSnapshot().turns[0].state).toBe('waiting')
})

it('cancels discarded speech and ignores late captions and finals for that id', async () => {
  const s = setup(); await s.start(); await s.emit(onset, audio, { type: 'discard' })
  expect(s.sockets[0].messages().at(-1)).toEqual({ type: 'cancel', id: 1 })
  s.sockets[0].message({ type: 'final', id: 1, result })
  expect(s.session.getSnapshot().turns[0].state).toBe('canceled')
  expect(s.play).not.toHaveBeenCalled()
  await s.emit(onset, audio, end)
  expect(s.session.getSnapshot().turns[1].id).toBe(2)
})

it('updates stable captions, replaces with final once, pauses only at playback, then listens again', async () => {
  const s = setup(); await s.start(); await s.emit(onset, audio, end)
  const ws = s.sockets[0]
  ws.message({ type: 'source', id: 1, text: '안녕 하', stable: 2 })
  ws.message({ type: 'translation', id: 1, text: 'Hello wor', stable: 0 })
  expect(s.session.getSnapshot().turns[0]).toMatchObject({ source: { text: '안녕 하', stable: 2 }, translation: { stable: 0 } })
  ws.message({ type: 'final', id: 1, result }); ws.message({ type: 'final', id: 1, result })
  expect(s.play).toHaveBeenCalledTimes(1)
  expect(s.session.getSnapshot().turns[0]).toMatchObject({ state: 'done', source: { text: '안녕하세요', stable: 5 } })
  await s.emit(onset, audio)
  expect(s.session.getSnapshot().speaking).toBe(true)
  s.played[0].begin()
  expect(ws.messages().at(-1)).toEqual({ type: 'cancel', id: 2 })
  expect(s.microphone.setPaused).toHaveBeenLastCalledWith(true)
  const count = s.detector.push.mock.calls.length
  s.capture(); expect(s.detector.push).toHaveBeenCalledTimes(count)
  s.played[0].task.resolve(); await vi.waitFor(() => expect(s.session.getSnapshot().playing).toBe(false))
  expect(s.microphone.setPaused).toHaveBeenLastCalledWith(false)
  await s.emit(onset, audio, end)
  expect(ws.messages().at(-1)).toEqual({ type: 'end', id: 3 })
})

it('reports a turn error safely and keeps processing the next utterance', async () => {
  const s = setup(); await s.start(); await s.emit(onset, audio, end)
  s.sockets[0].message({ type: 'error', id: 1, detail: 'No speech was recognized' })
  expect(s.session.getSnapshot().turns[0].error).toContain('말소리를 찾지 못했습니다')
  expect(s.session.getSnapshot().active).toBe(true)
  await s.emit(onset, audio, end)
  s.sockets[0].message({ type: 'error', id: 2, detail: 'private server exception' })
  expect(s.session.getSnapshot().turns[1].error).not.toContain('private')
  await s.emit(onset, audio, end)
  s.sockets[0].message({ type: 'final', id: 3, result })
  expect(s.session.getSnapshot().turns[2].state).toBe('done')
})

it('refreshes after 4401 only once, cancels old turns, resumes on ready and rejects stale socket events', async () => {
  const s = setup(); await s.start(); await s.emit(onset, audio, end)
  const stale = s.sockets[0].onmessage!
  s.sockets[0].closed(4401)
  expect(s.microphone.setPaused).toHaveBeenLastCalledWith(true)
  await vi.waitFor(() => expect(s.sockets).toHaveLength(2))
  expect(s.liveToken).toHaveBeenLastCalledWith('access')
  const ws = s.sockets[1]; ws.opened()
  expect(ws.messages()).toEqual([{ type: 'start', token: 'fresh', ...direction }])
  stale.call(s.sockets[0] as unknown as WebSocket, new MessageEvent('message', { data: JSON.stringify({ type: 'final', id: 1, result }) }))
  expect(s.play).not.toHaveBeenCalled()
  ws.message({ type: 'ready' })
  await s.emit(onset, audio, end)
  expect(ws.messages().at(-1)).toEqual({ type: 'end', id: 2 })
  ws.closed(4401)
  expect(s.liveToken).toHaveBeenCalledTimes(2)
  expect(s.logout).toHaveBeenCalledWith(true)
  expect(s.session.getSnapshot()).toMatchObject({ active: false, error: expect.stringContaining('로그인') })
})

it.each([4422, 4503, 1006])('stops on close %i and cleans the microphone without reconnecting', async (code) => {
  const s = setup(); await s.start(); await s.emit(onset, audio)
  s.sockets[0].closed(code)
  expect(s.session.getSnapshot()).toMatchObject({ active: false, error: expect.any(String) })
  expect(s.session.getSnapshot().error.length).toBeGreaterThan(0)
  expect(s.microphone.stop).toHaveBeenCalledTimes(1)
  expect(s.liveToken).toHaveBeenCalledTimes(1)
})

it('stops all pending speech, connection, playback and microphone, while preserving completed turns and replay', async () => {
  const s = setup(); await s.start(); await s.emit(onset, audio, end)
  s.sockets[0].message({ type: 'final', id: 1, result })
  await s.emit(onset, audio, end); await s.emit(onset, audio)
  s.session.stop()
  expect(s.sockets[0].messages().slice(-2)).toEqual([{ type: 'cancel', id: 3 }, { type: 'cancel', id: 2 }])
  expect(s.sockets[0].close).toHaveBeenCalledTimes(1)
  expect(s.played[0].signal.aborted).toBe(true)
  expect(s.microphone.stop).toHaveBeenCalledTimes(1)
  expect(s.session.getSnapshot().turns.map((turn) => turn.state)).toEqual(['done', 'canceled', 'canceled'])
  s.played[0].task.resolve(); await Promise.resolve()
  s.session.replay(1); expect(s.play).toHaveBeenCalledTimes(2)
})

it('drops detector work completing after playback interruption and microphone disconnect', async () => {
  const pending = deferred<LiveEndpointEvent[]>()
  const detector = { push: vi.fn(() => pending.promise), interrupt: vi.fn(() => true) }
  const s = setup(async () => detector); await s.start()
  s.capture(); s.lost(); pending.resolve([onset, audio, end]); await Promise.resolve()
  expect(s.session.getSnapshot().turns).toHaveLength(0)
  expect(s.session.getSnapshot().error).toContain('마이크')
})

it('stops safely while token refresh is pending without opening another socket', async () => {
  const s = setup(); await s.start()
  const pending = deferred<string>(); s.liveToken.mockReturnValueOnce(pending.promise)
  s.sockets[0].closed(4401); s.session.stop(); pending.resolve('late-token')
  await Promise.resolve(); await Promise.resolve()
  expect(s.sockets).toHaveLength(1)
})

it('times out a connection that never becomes ready', async () => {
  vi.useFakeTimers()
  const s = setup(); const starting = s.session.start(direction)
  await vi.advanceTimersByTimeAsync(12_001); await starting
  expect(s.session.getSnapshot()).toMatchObject({ active: false, error: expect.stringContaining('초과') })
  expect(s.sockets[0].close).toHaveBeenCalledTimes(1)
})

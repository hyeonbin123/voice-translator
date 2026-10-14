// [파일 역할] 동시통역 세션(LiveSession)과 웹소켓 전송 층(socket.ts의 LiveTransport)을 함께 검사하는 테스트
// 묶음. 실제 서버 없이 가짜 소켓으로 서버 메시지와 닫힘 코드를 일으킨다.
// [무엇을 지키나] 토큰은 주소가 아니라 첫 메시지에만 싣기, 메시지 순서와 PCM 값, 자막 갱신과 최종 결과 한
// 번만 반영, 마디 오류에서 서버 문자열 숨기기, 4401마다 새 토큰으로 재연결하되 무한 반복하지 않기,
// 닫힘·멈춤·시간 초과 때 자원 정리.
// [가짜 객체] 마이크·판정기·재생기는 테스트가 조종하는 가짜다. 판정기는 emit으로 정해 둔 이벤트를 다음
// push에서 그대로 돌려준다.

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
// 가짜 웹소켓. test/fakeSocket.ts와 같은 모양을 이 파일 안에 따로 둔다. opened·message·closed로 연결 열림,
// 서버 메시지, 닫힘을 일으키고, messages()로 보낸 것을 확인한다(소리는 'audio').
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
// 테스트마다 세션을 멈추고, 시간 초과 테스트가 켠 가짜 타이머를 되돌린다.
afterEach(() => { sessions.forEach((session) => session.stop()); sessions.length = 0; vi.useRealTimers() })

// 세션과 조작 도구. liveToken은 처음 'access', 거절된 토큰을 받으면 'fresh'를 준다. start는 소켓을 열고
// ready까지 받는다. emit은 다음 판정 결과를 정하고 마이크 조각 하나를 넣는다.
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

// 확인: 같은 호스트의 /api/translate/live로 붙고(https면 wss), 주소에는 토큰이 없으며 첫 start 메시지에만
// 토큰과 방향이 들어간다. ready 전에는 마이크 소리를 판정기에 넣지 않는다.
// 왜: 주소에 넣은 토큰은 서버·프록시 로그에 남는다(docs/api.md).
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

// 확인: 판정 이벤트가 utterance → 소리 → pause → (resume → 소리 → pause) → end 순서로 나가고, 소리 [−1, −0.5,
// 0, 0.5, 1]이 16비트 [−32768, −16384, 0, 16384, 32767]로 바뀌며, end 뒤 마디는 waiting이 된다.
it.each([false, true])('sends utterance/audio/pause/%s resume/pause/end in order with identical PCM16', async (resume) => {
  const s = setup(); await s.start()
  await s.emit(onset, audio, { type: 'pause' }, ...(resume ? [{ type: 'resume' }, audio, { type: 'pause' }] as LiveEndpointEvent[] : []), end)
  const expected = [{ type: 'utterance', id: 1 }, 'audio', { type: 'pause', id: 1 },
    ...(resume ? [{ type: 'resume', id: 1 }, 'audio', { type: 'pause', id: 1 }] : []), { type: 'end', id: 1 }]
  expect(s.sockets[0].messages().slice(1)).toEqual(expected)
  expect([...new Int16Array(s.sockets[0].send.mock.calls[2][0] as ArrayBuffer)]).toEqual([-32768, -16384, 0, 16384, 32767])
  expect(s.session.getSnapshot().turns[0].state).toBe('waiting')
})

// 확인: 짧은 소리로 버린 마디는 cancel을 보내고, 그 번호로 늦게 온 final은 무시하며 재생하지 않는다. 다음
// 마디는 새 번호(2)를 쓴다.
it('cancels discarded speech and ignores late captions and finals for that id', async () => {
  const s = setup(); await s.start(); await s.emit(onset, audio, { type: 'discard' })
  expect(s.sockets[0].messages().at(-1)).toEqual({ type: 'cancel', id: 1 })
  s.sockets[0].message({ type: 'final', id: 1, result })
  expect(s.session.getSnapshot().turns[0].state).toBe('canceled')
  expect(s.play).not.toHaveBeenCalled()
  await s.emit(onset, audio, end)
  expect(s.session.getSnapshot().turns[1].id).toBe(2)
})

// 확인: source·translation 자막과 stable이 그대로 반영되고, final은 같은 것이 두 번 와도 한 번만 재생되며
// 자막을 최종 원문으로 모두 진하게 바꾼다. 재생이 실제로 시작될 때 말하던 마디를 취소하고 마이크를 쉬게 하며,
// 재생이 끝나면 다시 듣는다.
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

// 확인: 마디 오류는 그 마디에만 한국어 안내로 표시하고 동시통역은 계속된다. 계약에 없는 detail('private
// server exception')은 화면에 그대로 나오지 않는다.
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

// 확인: ready 뒤 4401로 닫힐 때마다 거절된 토큰을 넘겨 새 토큰으로 다시 연결하고, 로그아웃하지 않는다. 옛
// 소켓에서 늦게 온 final은 무시한다. 마디 번호는 새 연결에서도 이어진다.
// 왜: 예전에는 재연결 기회가 세션 전체에 한 번뿐이라, 두 번째 만료(30~60분 뒤)에 아직 유효한 refresh 토큰을
// 지우고 로그인 화면으로 보냈다(c90d1ee).
it('refreshes and reconnects on each 4401 after ready, cancels old turns and rejects stale socket events', async () => {
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
  // 다음 접근 토큰도 연결 도중 만료된다. 서버는 그때마다 4401로 닫는다.
  ws.closed(4401)
  await vi.waitFor(() => expect(s.sockets).toHaveLength(3))
  expect(s.liveToken).toHaveBeenCalledTimes(3)
  expect(s.liveToken).toHaveBeenLastCalledWith('fresh')
  expect(s.logout).not.toHaveBeenCalled()
  expect(s.session.getSnapshot().active).toBe(true)
  const next = s.sockets[2]; next.opened(); next.message({ type: 'ready' })
  expect(s.session.getSnapshot()).toMatchObject({ active: true, ready: true })
  await s.emit(onset, audio, end)
  expect(next.messages().at(-1)).toEqual({ type: 'end', id: 3 })
})

// 확인: 새로 받은 토큰마저 ready 전에 4401로 거절되면 더 연결하지 않고 만료 로그아웃(logout(true))하며 로그인
// 안내를 띄운다. 거절이 끝없이 재연결로 이어지지 않게 하는 안전장치다.
it('logs out when the refreshed token is rejected again before ready', async () => {
  const s = setup(); const starting = s.session.start(direction)
  await vi.waitFor(() => expect(s.sockets).toHaveLength(1))
  s.sockets[0].opened(); s.sockets[0].closed(4401)
  await vi.waitFor(() => expect(s.sockets).toHaveLength(2))
  s.sockets[1].opened(); s.sockets[1].closed(4401)
  await starting
  expect(s.sockets).toHaveLength(2)
  expect(s.liveToken).toHaveBeenCalledTimes(2)
  expect(s.logout).toHaveBeenCalledWith(true)
  expect(s.session.getSnapshot()).toMatchObject({ active: false, error: expect.stringContaining('로그인') })
})

// 확인: 4422(요청 오류), 4503(모델 없음), 1006(비정상 끊김)으로 닫히면 다시 연결하지 않고 멈추며 안내를
// 띄우고 마이크를 닫는다.
it.each([4422, 4503, 1006])('stops on close %i and cleans the microphone without reconnecting', async (code) => {
  const s = setup(); await s.start(); await s.emit(onset, audio)
  s.sockets[0].closed(code)
  expect(s.session.getSnapshot()).toMatchObject({ active: false, error: expect.any(String) })
  expect(s.session.getSnapshot().error.length).toBeGreaterThan(0)
  expect(s.microphone.stop).toHaveBeenCalledTimes(1)
  expect(s.liveToken).toHaveBeenCalledTimes(1)
})

// 확인: 멈추면 말하던 마디(3)와 결과를 기다리던 마디(2)에 cancel을 보내고, 소켓·재생·마이크를 닫는다. 끝난
// 마디(1)는 남고 멈춘 뒤에도 다시 듣기가 된다.
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

// 확인: 판정 결과가 늦게 나오는 사이 마이크가 끊기면, 그 결과로 마디를 만들지 않고 마이크 오류만 보인다.
it('drops detector work completing after playback interruption and microphone disconnect', async () => {
  const pending = deferred<LiveEndpointEvent[]>()
  const detector = { push: vi.fn(() => pending.promise), interrupt: vi.fn(() => true) }
  const s = setup(async () => detector); await s.start()
  s.capture(); s.lost(); pending.resolve([onset, audio, end]); await Promise.resolve()
  expect(s.session.getSnapshot().turns).toHaveLength(0)
  expect(s.session.getSnapshot().error).toContain('마이크')
})

// 확인: 4401 뒤 새 토큰을 기다리는 중에 멈추면, 늦게 받은 토큰으로 새 소켓을 열지 않는다.
it('stops safely while token refresh is pending without opening another socket', async () => {
  const s = setup(); await s.start()
  const pending = deferred<string>(); s.liveToken.mockReturnValueOnce(pending.promise)
  s.sockets[0].closed(4401); s.session.stop(); pending.resolve('late-token')
  await Promise.resolve(); await Promise.resolve()
  expect(s.sockets).toHaveLength(1)
})

// 확인: 서버가 12초 안에 ready를 보내지 않으면 시간 초과 안내와 함께 멈추고 소켓을 닫는다. 가짜 타이머로
// 시간을 앞당긴다.
it('times out a connection that never becomes ready', async () => {
  vi.useFakeTimers()
  const s = setup(); const starting = s.session.start(direction)
  await vi.advanceTimersByTimeAsync(12_001); await starting
  expect(s.session.getSnapshot()).toMatchObject({ active: false, error: expect.stringContaining('초과') })
  expect(s.sockets[0].close).toHaveBeenCalledTimes(1)
})

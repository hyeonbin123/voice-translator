import { afterEach, expect, it, vi } from 'vitest'
import { TranslationApi, type TranslationResult } from '../translation/api'
import { ConversationSession } from './session'
import { SileroEndpointer, type CreateEndpointer, type LiveEndpointEvent } from './silero'
import type { OpenMicrophone } from './microphone'
import type { PlayTranslation } from './player'
import { FakeSocket } from '../test/fakeSocket'

const direction = { source_lang: 'ko', target_lang: 'en' } as const
const result = (id: string): TranslationResult => ({ id, mode: 'speech', ...direction,
  source_text: `원문 ${id}`, translated_text: `Translation ${id}`, audio_id: null, tts_error: null,
  stt_model: 'stt', mt_model: 'mt', tts_model: null, stt_ms: 10, mt_ms: 10, tts_ms: null, created_at: '' })
function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (error: unknown) => void
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no })
  return { promise, resolve, reject }
}
const utterance = () => {
  const pcm = new Float32Array(39 * 512)
  pcm.fill(.75, 0, 8 * 512)
  return pcm
}
const sessions: ConversationSession[] = []
afterEach(() => { sessions.forEach((session) => session.stop()); sessions.length = 0 })

function setup(create?: CreateEndpointer) {
  let capture!: (samples: Float32Array) => void
  let disconnected!: () => void
  const microphone = { stop: vi.fn(), setPaused: vi.fn() }
  const open = vi.fn<OpenMicrophone>(async (samples, _signal, failed) => {
    capture = samples; disconnected = failed; return microphone
  })
  const requests: ReturnType<typeof deferred<Response>>[] = []
  const request = vi.fn<(path: string, init?: RequestInit) => Promise<Response>>(() => {
    const task = deferred<Response>(); requests.push(task); return task.promise
  })
  const played: { id: string; begin: () => void; task: ReturnType<typeof deferred<void>>; signal: AbortSignal }[] = []
  const play = vi.fn<PlayTranslation>((turn, signal, begin) => {
    const task = deferred<void>(); played.push({ id: turn.id, begin, task, signal }); return task.promise
  })
  const run = vi.fn(async (input: Float32Array) => ({ probability: input[64], h: new Float32Array(128), c: new Float32Array(128) }))
  // Example mode (no login, VITE_TRANSLATION_MOCK) uploads each clip over HTTP after its end.
  const session = new ConversationSession(new TranslationApi({ request }, true), open, play,
    create ?? (async () => new SileroEndpointer(run)))
  sessions.push(session)
  const send = async (count: number) => {
    capture(utterance())
    await vi.waitFor(() => expect(session.getSnapshot().turns).toHaveLength(count))
  }
  const reply = (i: number) => requests[i].resolve(new Response(JSON.stringify(result(String(i + 1))), { status: 201 }))
  return { session, microphone, open, request, played, play, run, send, reply, requests,
    capture: (samples: Float32Array) => capture(samples), disconnect: () => disconnected() }
}

it('in example mode listens during translation, uploads WAV over HTTP, and sends one request at a time', async () => {
  const s = setup(); await s.session.start(direction)
  await s.send(1); await s.send(2)
  expect(s.request).toHaveBeenCalledTimes(1)
  expect(s.session.getSnapshot()).toMatchObject({ active: true, translating: true, playing: false })
  const [path, init] = s.request.mock.calls[0]
  expect(path).toBe('/api/translate/speech')
  const form = init!.body as FormData
  expect(form.get('source_lang')).toBe('ko'); expect(form.get('target_lang')).toBe('en')
  const wav = form.get('audio') as File
  expect(wav.name).toBe('conversation.wav'); expect(wav.type).toBe('audio/wav')
  expect(wav.size).toBe(44 + (8 + 6) * 512 * 2)
  s.reply(0)
  await vi.waitFor(() => expect(s.request).toHaveBeenCalledTimes(2))
  s.reply(1)
  await vi.waitFor(() => expect(s.session.getSnapshot().turns.every((t) => t.state === 'done')).toBe(true))
  expect(s.played.map((p) => p.id)).toEqual(['1'])
  s.played[0].begin()
  expect(s.microphone.setPaused).toHaveBeenLastCalledWith(true)
  const before = s.run.mock.calls.length
  s.capture(utterance())
  await Promise.resolve()
  expect(s.run).toHaveBeenCalledTimes(before)
  s.played[0].task.resolve()
  await vi.waitFor(() => expect(s.played.map((p) => p.id)).toEqual(['1', '2']))
  s.played[1].begin(); s.played[1].task.resolve()
  await vi.waitFor(() => expect(s.session.getSnapshot().playingId).toBeNull())
  expect(s.microphone.setPaused).toHaveBeenLastCalledWith(false)
  await s.send(3)
  expect(s.request).toHaveBeenCalledTimes(3)
})

it('keeps listening while audio is downloading, then discards unfinished speech at playback', async () => {
  const s = setup(); await s.session.start(direction); await s.send(1); s.reply(0)
  await vi.waitFor(() => expect(s.played).toHaveLength(1))
  s.capture(new Float32Array(2 * 512).fill(.8))
  await vi.waitFor(() => expect(s.session.getSnapshot().speaking).toBe(true))
  s.played[0].begin()
  expect(s.session.getSnapshot()).toMatchObject({ playing: true, speaking: false })
  expect(s.session.getSnapshot().notice).toContain('아직 끝나지 않은 말')
})

it('shows a single-turn API error and continues with the queued next turn', async () => {
  const s = setup(); await s.session.start(direction); await s.send(1); await s.send(2)
  s.requests[0].reject(new Error('failed'))
  await vi.waitFor(() => expect(s.request).toHaveBeenCalledTimes(2))
  expect(s.session.getSnapshot().turns[0]).toMatchObject({ state: 'error' })
  s.reply(1)
  await vi.waitFor(() => expect(s.session.getSnapshot().turns[1].state).toBe('done'))
  expect(s.session.getSnapshot().active).toBe(true)
})

it('stop aborts requests and playback, cancels queued turns and ignores stale responses after restart', async () => {
  const s = setup(); await s.session.start(direction); await s.send(1); await s.send(2)
  const signal = s.request.mock.calls[0][1]!.signal!
  s.session.stop()
  expect(signal.aborted).toBe(true); expect(s.microphone.stop).toHaveBeenCalledOnce()
  expect(s.session.getSnapshot().turns.map((t) => t.state)).toEqual(['canceled', 'canceled'])
  await s.session.start(direction)
  s.reply(0)
  await Promise.resolve(); await Promise.resolve()
  expect(s.play).not.toHaveBeenCalled()
  await s.send(3); s.reply(1)
  await vi.waitFor(() => expect(s.played).toHaveLength(1))
  s.played[0].begin(); s.session.stop()
  expect(s.played[0].signal.aborted).toBe(true)
  s.played[0].task.resolve()
  await Promise.resolve()
  expect(s.session.getSnapshot()).toMatchObject({ active: false, playing: false, translating: false })
})

it('playback failure resumes listening and supports replay after stopping', async () => {
  const s = setup(); await s.session.start(direction); await s.send(1); s.reply(0)
  await vi.waitFor(() => expect(s.played).toHaveLength(1))
  s.played[0].begin(); s.played[0].task.reject(new Error('autoplay denied'))
  await vi.waitFor(() => expect(s.session.getSnapshot().turns[0].audioError).toBeTruthy())
  expect(s.session.getSnapshot()).toMatchObject({ active: true, playing: false })
  s.session.stop(); s.session.replay(1)
  await vi.waitFor(() => expect(s.played).toHaveLength(2))
  s.played[1].begin(); s.session.stop()
  expect(s.played[1].signal.aborted).toBe(true)
})

it('stop during lazy model loading releases microphone and ignores the late detector', async () => {
  const load = deferred<SileroEndpointer>()
  const s = setup(() => load.promise)
  const started = s.session.start(direction)
  await vi.waitFor(() => expect(s.open).toHaveBeenCalledOnce())
  s.capture(utterance())
  expect(s.request).not.toHaveBeenCalled()
  s.session.stop()
  load.resolve(new SileroEndpointer(s.run))
  await started
  expect(s.session.getSnapshot()).toMatchObject({ active: false, permission: false })
  expect(s.microphone.stop).toHaveBeenCalledOnce()
})

it('model load failure stops capture and permits another start', async () => {
  const create = vi.fn<CreateEndpointer>().mockRejectedValueOnce(new Error('load failed'))
    .mockResolvedValue(new SileroEndpointer(async () => ({ probability: 0, h: new Float32Array(128), c: new Float32Array(128) })))
  const s = setup(create); await s.session.start(direction)
  expect(s.session.getSnapshot().error).toBeTruthy()
  expect(s.microphone.stop).toHaveBeenCalledOnce()
  await s.session.start(direction)
  expect(s.session.getSnapshot()).toMatchObject({ active: true, error: '' })
})

it('reports disconnected microphones and discards a short noise burst', async () => {
  const s = setup(); await s.session.start(direction)
  const pcm = new Float32Array(38 * 512); pcm.fill(.9, 0, 7 * 512)
  s.capture(pcm)
  await vi.waitFor(() => expect(s.session.getSnapshot().notice).toContain('짧은 소리'))
  expect(s.request).not.toHaveBeenCalled()
  s.disconnect()
  expect(s.session.getSnapshot()).toMatchObject({ active: false })
  expect(s.session.getSnapshot().error).toContain('연결이 끊겼습니다')
})

// Signed in, the clip streams over the live connection without subtitles and the server prepares its
// translation at a 192 ms pause (T77, docs/experiments.md 11).
function streaming() {
  let capture!: (samples: Float32Array) => void
  const microphone = { stop: vi.fn(), setPaused: vi.fn() }
  const open = vi.fn<OpenMicrophone>(async (handler) => { capture = handler; return microphone })
  let events: LiveEndpointEvent[] = []
  const detector = { interrupt: vi.fn(() => true), push: vi.fn(async () => events) }
  const sockets: FakeSocket[] = []
  const request = vi.fn<(path: string, init?: RequestInit) => Promise<Response>>()
  const liveToken = vi.fn(async (rejected?: string): Promise<string> => rejected ? 'fresh' : 'access')
  const api = new TranslationApi({ request, liveToken, logout: vi.fn() })
  const played: { id: string; begin: () => void; task: ReturnType<typeof deferred<void>> }[] = []
  const play = vi.fn<PlayTranslation>((turn, _signal, begin) => {
    const task = deferred<void>(); played.push({ id: turn.id, begin, task }); return task.promise
  })
  const session = new ConversationSession(api, open, play, async () => detector, () => {
    const socket = new FakeSocket(); sockets.push(socket); return socket
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
  return { session, sockets, microphone, detector, request, liveToken, played, play, start, emit,
    capture: () => capture(new Float32Array(512)) }
}
const onset: LiveEndpointEvent = { type: 'start', startSample: 0 }
const audio: LiveEndpointEvent = { type: 'audio', samples: new Float32Array([-1, 0, 1]) }
const end: LiveEndpointEvent = { type: 'end', samples: new Float32Array(512), startSample: 0, endSample: 512, detectedAtSample: 0, forced: false }

it('streams each utterance, prepares at pauses and commits at the end without an upload', async () => {
  const s = streaming(); const starting = s.session.start(direction)
  await vi.waitFor(() => expect(s.sockets).toHaveLength(1))
  s.capture(); expect(s.detector.push).not.toHaveBeenCalled()
  s.sockets[0].opened()
  expect(s.sockets[0].messages()).toEqual([{ type: 'start', token: 'access', ...direction, mode: 'conversation' }])
  s.sockets[0].message({ type: 'ready' }); await starting
  expect(s.microphone.setPaused).toHaveBeenLastCalledWith(false)
  await s.emit(onset, audio, { type: 'pause' }, { type: 'resume' }, audio, { type: 'pause' }, end)
  expect(s.sockets[0].messages().slice(1)).toEqual([{ type: 'utterance', id: 1 }, 'audio', { type: 'pause', id: 1 },
    { type: 'resume', id: 1 }, 'audio', { type: 'pause', id: 1 }, { type: 'end', id: 1 }])
  expect(s.session.getSnapshot()).toMatchObject({ translating: true, turns: [{ id: 1, state: 'translating' }] })
  s.sockets[0].message({ type: 'final', id: 1, result: result('1') })
  await vi.waitFor(() => expect(s.played.map((p) => p.id)).toEqual(['1']))
  expect(s.session.getSnapshot().turns[0]).toMatchObject({ state: 'done', result: { id: '1' } })
  expect(s.request).not.toHaveBeenCalled()
})

it('keeps turns in the order spoken when finals come out of order, with an error on its own turn', async () => {
  const s = streaming(); await s.start()
  await s.emit(onset, audio, end); await s.emit(onset, audio, end); await s.emit(onset, audio, end)
  s.sockets[0].message({ type: 'final', id: 2, result: result('2') })
  await Promise.resolve(); await Promise.resolve()
  expect(s.played).toHaveLength(0)
  s.sockets[0].message({ type: 'error', id: 1, detail: 'No speech was recognized' })
  await vi.waitFor(() => expect(s.played.map((p) => p.id)).toEqual(['2']))
  expect(s.session.getSnapshot().turns[0]).toMatchObject({ state: 'error', error: '말소리를 찾지 못했습니다. 마이크를 확인해 주세요.' })
  s.sockets[0].message({ type: 'final', id: 3, result: result('3') })
  s.played[0].begin(); s.played[0].task.resolve()
  await vi.waitFor(() => expect(s.played.map((p) => p.id)).toEqual(['2', '3']))
})

it('cancels speech in progress on the server when playback starts and listens again after it', async () => {
  const s = streaming(); await s.start(); await s.emit(onset, audio, end)
  s.sockets[0].message({ type: 'final', id: 1, result: result('1') })
  await vi.waitFor(() => expect(s.played).toHaveLength(1))
  await s.emit(onset, audio)
  s.played[0].begin()
  expect(s.sockets[0].messages()).toContainEqual({ type: 'cancel', id: 2 })
  expect(s.session.getSnapshot()).toMatchObject({ playing: true, speaking: false })
  expect(s.session.getSnapshot().notice).toContain('아직 끝나지 않은 말')
  expect(s.microphone.setPaused).toHaveBeenLastCalledWith(true)
  s.played[0].task.resolve()
  await vi.waitFor(() => expect(s.microphone.setPaused).toHaveBeenLastCalledWith(false))
  await s.emit(end)
  expect(s.sockets[0].messages()).not.toContainEqual({ type: 'end', id: 2 })
  expect(s.session.getSnapshot().turns).toHaveLength(1)
})

it('reconnects with a new token, cancelling turns still waiting and the speech in progress', async () => {
  const s = streaming(); await s.start(); await s.emit(onset, audio, end); await s.emit(onset, audio)
  s.sockets[0].closed(4401)
  await vi.waitFor(() => expect(s.session.getSnapshot().turns[0].state).toBe('canceled'))
  expect(s.session.getSnapshot()).toMatchObject({ active: true, speaking: false, translating: false })
  expect(s.session.getSnapshot().notice).toContain('다시 말해 주세요')
  expect(s.detector.interrupt).toHaveBeenCalled()
  expect(s.microphone.setPaused).toHaveBeenLastCalledWith(true)
  await vi.waitFor(() => expect(s.sockets).toHaveLength(2))
  expect(s.liveToken).toHaveBeenLastCalledWith('access')
  s.sockets[1].opened()
  expect(s.sockets[1].messages()).toEqual([{ type: 'start', token: 'fresh', ...direction, mode: 'conversation' }])
  s.sockets[1].message({ type: 'ready' })
  expect(s.microphone.setPaused).toHaveBeenLastCalledWith(false)
  await s.emit(onset, audio, end)
  expect(s.sockets[1].messages()).toContainEqual({ type: 'end', id: 3 })
})

it('stop closes the connection, cancels waiting turns and ignores what comes after', async () => {
  const s = streaming(); await s.start(); await s.emit(onset, audio, end)
  s.session.stop()
  expect(s.sockets[0].close).toHaveBeenCalled()
  expect(s.session.getSnapshot()).toMatchObject({ active: false, translating: false, turns: [{ id: 1, state: 'canceled' }] })
  s.sockets[0].message({ type: 'final', id: 1, result: result('1') })
  await Promise.resolve()
  expect(s.play).not.toHaveBeenCalled()
})

it('names conversation mode when its connection fails', async () => {
  const s = streaming(); await s.start()
  s.sockets[0].closed(4503)
  expect(s.session.getSnapshot()).toMatchObject({ active: false,
    error: '대화 모드 서비스를 사용할 수 없습니다. 잠시 후 다시 시작해 주세요.' })
})

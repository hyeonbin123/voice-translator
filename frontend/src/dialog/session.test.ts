import { afterEach, expect, it, vi } from 'vitest'
import { TranslationApi, type DialogTranslationResult, type TranslationResult } from '../translation/api'
import { SileroEndpointer, type CreateEndpointer, type LiveEndpointEvent } from '../conversation/silero'
import type { OpenMicrophone } from '../conversation/microphone'
import type { PlayTranslation } from '../conversation/player'
import { DialogSession } from './session'
import { FakeSocket } from '../test/fakeSocket'

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
const speechResult = (id: string, source_lang: 'ko' | 'en'): TranslationResult => ({
  id, mode: 'speech', source_lang, target_lang: source_lang === 'ko' ? 'en' : 'ko',
  source_text: source_lang === 'ko' ? `원문 ${id}` : `Source ${id}`,
  translated_text: source_lang === 'ko' ? `Translation ${id}` : `번역 ${id}`,
  audio_id: null, tts_error: null, stt_model: 'stt', mt_model: 'mt', tts_model: null,
  stt_ms: 10, mt_ms: 10, tts_ms: null, created_at: '',
})
const result = (id: string, source_lang: 'ko' | 'en', guessed = false): DialogTranslationResult => ({
  ...speechResult(id, source_lang), language_confidence: guessed ? .55 : .98,
  language_guessed: guessed,
})
const sessions: DialogSession[] = []
afterEach(() => { sessions.forEach((session) => session.stop()); sessions.length = 0 })

function setup(create?: CreateEndpointer) {
  let capture!: (samples: Float32Array) => void
  const microphone = { stop: vi.fn(), setPaused: vi.fn() }
  const open = vi.fn<OpenMicrophone>(async (samples) => { capture = samples; return microphone })
  const requests: ReturnType<typeof deferred<Response>>[] = []
  const request = vi.fn<(path: string, init?: RequestInit) => Promise<Response>>(() => {
    const task = deferred<Response>(); requests.push(task); return task.promise
  })
  const played: { result: TranslationResult; begin: () => void; task: ReturnType<typeof deferred<void>>; signal: AbortSignal }[] = []
  const play = vi.fn<PlayTranslation>((translation, signal, begin) => {
    const task = deferred<void>(); played.push({ result: translation, begin, task, signal }); return task.promise
  })
  const run = vi.fn(async (input: Float32Array) => ({ probability: input[64], h: new Float32Array(128), c: new Float32Array(128) }))
  // Example mode (no login, VITE_TRANSLATION_MOCK) uploads each turn over HTTP after its end.
  const session = new DialogSession(new TranslationApi({ request }, true), open, play,
    create ?? (async () => new SileroEndpointer(run)))
  sessions.push(session)
  const send = async (count: number) => {
    capture(utterance())
    await vi.waitFor(() => expect(session.getSnapshot().turns).toHaveLength(count))
  }
  const reply = (index: number, body: TranslationResult | DialogTranslationResult) =>
    requests[index].resolve(new Response(JSON.stringify(body), { status: 201 }))
  return { session, microphone, request, requests, played, run, send, reply,
    capture: (samples: Float32Array) => capture(samples) }
}

it('uploads the same WAV sequentially and sends only the last processed source language', async () => {
  const s = setup(); await s.session.start()
  await s.send(1); await s.send(2)
  expect(s.request).toHaveBeenCalledTimes(1)
  expect(s.request.mock.calls[0][0]).toBe('/api/translate/dialog')
  const first = s.request.mock.calls[0][1]!.body as FormData
  expect(first.get('previous_lang')).toBeNull()
  expect(first.get('audio')).toMatchObject({ name: 'conversation.wav', type: 'audio/wav' })
  s.reply(0, result('one', 'ko'))
  await vi.waitFor(() => expect(s.request).toHaveBeenCalledTimes(2))
  const second = s.request.mock.calls[1][1]!.body as FormData
  expect(second.get('previous_lang')).toBe('ko')
  s.reply(1, result('two', 'en'))
  await vi.waitFor(() => expect(s.session.getSnapshot().turns.every((turn) => turn.state === 'done')).toBe(true))

  s.session.stop(); await s.session.start(); await s.send(3)
  const restarted = s.request.mock.calls[2][1]!.body as FormData
  expect(restarted.get('previous_lang')).toBeNull()
})

it('pauses capture only while translated audio is playing and cleans up on stop', async () => {
  const s = setup(); await s.session.start(); await s.send(1)
  s.reply(0, result('one', 'ko'))
  await vi.waitFor(() => expect(s.played).toHaveLength(1))
  const before = s.run.mock.calls.length
  s.played[0].begin()
  expect(s.microphone.setPaused).toHaveBeenLastCalledWith(true)
  s.capture(utterance()); await Promise.resolve()
  expect(s.run).toHaveBeenCalledTimes(before)
  s.played[0].task.resolve()
  await vi.waitFor(() => expect(s.microphone.setPaused).toHaveBeenLastCalledWith(false))
  expect(s.session.getSnapshot()).toMatchObject({ active: true, playing: false })
  s.session.stop()
  expect(s.microphone.stop).toHaveBeenCalledOnce()
  expect(s.session.hasRecording(1)).toBe(false)
  expect(s.played[0].signal.aborted).toBe(true)
})

it('re-sends a guessed turn in the opposite direction, deletes the wrong record, and replaces it', async () => {
  const s = setup(); await s.session.start(); await s.send(1)
  const original = result('wrong', 'ko', true)
  s.reply(0, original)
  await vi.waitFor(() => expect(s.played).toHaveLength(1))
  s.played[0].begin(); s.played[0].task.resolve()
  await vi.waitFor(() => expect(s.session.getSnapshot().playingId).toBeNull())

  const correcting = s.session.reverse(1)
  await vi.waitFor(() => expect(s.request).toHaveBeenCalledTimes(2))
  expect(s.request.mock.calls[1][0]).toBe('/api/translate/speech')
  const form = s.request.mock.calls[1][1]!.body as FormData
  expect(form.get('source_lang')).toBe('en'); expect(form.get('target_lang')).toBe('ko')
  const replacement = speechResult('right', 'en')
  s.reply(1, replacement)
  await vi.waitFor(() => expect(s.request).toHaveBeenCalledTimes(3))
  expect(s.request.mock.calls[2][0]).toBe('/api/history/wrong')
  expect(s.request.mock.calls[2][1]?.method).toBe('DELETE')
  expect(s.session.getSnapshot().turns[0].result).toEqual(replacement)
  s.requests[2].resolve(new Response(null, { status: 204 }))
  await correcting
  expect(s.session.getSnapshot().turns[0].result).toMatchObject({ id: 'right', source_lang: 'en' })
  await vi.waitFor(() => expect(s.played).toHaveLength(2))
})

it('keeps the original guessed bubble when retranslation fails', async () => {
  const s = setup(); await s.session.start(); await s.send(1)
  const original = result('wrong', 'ko', true)
  s.reply(0, original)
  await vi.waitFor(() => expect(s.played).toHaveLength(1))
  s.played[0].begin(); s.played[0].task.resolve()
  await vi.waitFor(() => expect(s.session.getSnapshot().playingId).toBeNull())

  const retry = s.session.reverse(1)
  await vi.waitFor(() => expect(s.request).toHaveBeenCalledTimes(2))
  s.requests[1].reject(new Error('retry failed')); await retry
  expect(s.session.getSnapshot().turns[0]).toMatchObject({
    result: original,
    correctionError: expect.stringContaining('원래 말풍선을 유지했습니다.'),
  })
})

it('shows the replacement and tells the user when deleting the wrong record fails', async () => {
  const s = setup(); await s.session.start(); await s.send(1)
  const original = result('wrong', 'ko', true)
  s.reply(0, original)
  await vi.waitFor(() => expect(s.played).toHaveLength(1))
  s.played[0].begin(); s.played[0].task.resolve()
  await vi.waitFor(() => expect(s.session.getSnapshot().playingId).toBeNull())

  const deleteFailure = s.session.reverse(1)
  await vi.waitFor(() => expect(s.request).toHaveBeenCalledTimes(2))
  const replacement = speechResult('right', 'en')
  s.reply(1, replacement)
  await vi.waitFor(() => expect(s.request).toHaveBeenCalledTimes(3))
  expect(s.session.getSnapshot().turns[0].result).toEqual(replacement)
  s.requests[2].reject(new Error('delete failed')); await deleteFailure
  expect(s.session.getSnapshot().turns[0]).toMatchObject({
    result: replacement,
    correctionError: '새 번역은 반영했지만 예전 기록을 지우지 못했습니다. 기록 화면에서 지울 수 있습니다.',
  })
})

// Signed in, each turn streams over the live connection in dialog mode: the server detects its language and
// prepares it at a 192 ms pause, and keeps the last turn's language itself (T77, docs/experiments.md 11).
function streaming() {
  let capture!: (samples: Float32Array) => void
  const microphone = { stop: vi.fn(), setPaused: vi.fn() }
  const open = vi.fn<OpenMicrophone>(async (handler) => { capture = handler; return microphone })
  let events: LiveEndpointEvent[] = []
  const detector = { interrupt: vi.fn(() => true), push: vi.fn(async () => events) }
  const sockets: FakeSocket[] = []
  const requests: ReturnType<typeof deferred<Response>>[] = []
  const request = vi.fn<(path: string, init?: RequestInit) => Promise<Response>>(() => {
    const task = deferred<Response>(); requests.push(task); return task.promise
  })
  const liveToken = vi.fn(async (rejected?: string): Promise<string> => rejected ? 'fresh' : 'access')
  const api = new TranslationApi({ request, liveToken, logout: vi.fn() })
  const played: { result: TranslationResult; begin: () => void; task: ReturnType<typeof deferred<void>> }[] = []
  const play = vi.fn<PlayTranslation>((translation, _signal, begin) => {
    const task = deferred<void>(); played.push({ result: translation, begin, task }); return task.promise
  })
  const session = new DialogSession(api, open, play, async () => detector, () => {
    const socket = new FakeSocket(); sockets.push(socket); return socket
  })
  sessions.push(session)
  const start = async () => {
    const starting = session.start()
    await vi.waitFor(() => expect(sockets).toHaveLength(1))
    sockets[0].opened(); sockets[0].message({ type: 'ready' }); await starting
  }
  const emit = async (...next: LiveEndpointEvent[]) => {
    events = next; capture(new Float32Array(512)); await Promise.resolve(); await Promise.resolve()
  }
  return { session, sockets, request, requests, played, start, emit }
}
const onset: LiveEndpointEvent = { type: 'start', startSample: 0 }
const audio: LiveEndpointEvent = { type: 'audio', samples: new Float32Array([-1, 0, 1]) }
const end: LiveEndpointEvent = { type: 'end', samples: new Float32Array(512), startSample: 0, endSample: 512, detectedAtSample: 0, forced: false }

it('streams turns in dialog mode and starts a new connection from the last turn it knows', async () => {
  const s = streaming(); await s.start()
  expect(s.sockets[0].messages()[0]).toEqual({ type: 'start', token: 'access', mode: 'dialog' })
  await s.emit(onset, audio, { type: 'pause' }, end)
  expect(s.sockets[0].messages().slice(1)).toEqual([{ type: 'utterance', id: 1 }, 'audio', { type: 'pause', id: 1 }, { type: 'end', id: 1 }])
  s.sockets[0].message({ type: 'final', id: 1, result: speechResult('plain', 'ko') }) // no language fields: not a dialog answer
  s.sockets[0].message({ type: 'final', id: 1, result: result('one', 'ko') })
  await vi.waitFor(() => expect(s.session.getSnapshot().turns[0]).toMatchObject({ state: 'done', result: { id: 'one' } }))
  expect(s.request).not.toHaveBeenCalled()
  s.sockets[0].closed(4401)
  await vi.waitFor(() => expect(s.sockets).toHaveLength(2))
  s.sockets[1].opened()
  expect(s.sockets[1].messages()[0]).toEqual({ type: 'start', token: 'fresh', mode: 'dialog', previous_lang: 'ko', previous_id: 1 })
})

it('tells the server when the last turn is turned around, and not for an older one', async () => {
  const s = streaming(); await s.start()
  const previous = () => s.sockets[0].messages().filter((m) => (m as { type?: string }).type === 'previous')
  const listen = async (count: number) => {
    await vi.waitFor(() => expect(s.played).toHaveLength(count))
    s.played[count - 1].begin(); s.played[count - 1].task.resolve()
    await vi.waitFor(() => expect(s.session.getSnapshot().playingId).toBeNull())
  }
  await s.emit(onset, audio, end)
  s.sockets[0].message({ type: 'final', id: 1, result: result('first', 'ko', true) })
  await listen(1)
  await s.emit(onset, audio, end)
  s.sockets[0].message({ type: 'final', id: 2, result: result('second', 'en', true) })
  await listen(2)

  const older = s.session.reverse(1)
  await vi.waitFor(() => expect(s.request).toHaveBeenCalledTimes(1))
  expect(s.request.mock.calls[0][0]).toBe('/api/translate/speech')
  s.requests[0].resolve(new Response(JSON.stringify(speechResult('first-fixed', 'en')), { status: 201 }))
  await vi.waitFor(() => expect(s.request).toHaveBeenCalledTimes(2))
  s.requests[1].resolve(new Response(null, { status: 204 })); await older
  await listen(3)
  expect(previous()).toEqual([])

  const last = s.session.reverse(2)
  await vi.waitFor(() => expect(s.request).toHaveBeenCalledTimes(3))
  s.requests[2].resolve(new Response(JSON.stringify(speechResult('second-fixed', 'ko')), { status: 201 }))
  await vi.waitFor(() => expect(s.request).toHaveBeenCalledTimes(4))
  s.requests[3].resolve(new Response(null, { status: 204 })); await last
  expect(previous()).toEqual([{ type: 'previous', id: 2, lang: 'ko' }])
})

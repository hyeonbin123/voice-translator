// [파일 역할] 두 사람 대화 세션(DialogSession) 테스트 묶음. 앞부분은 예시 모드(HTTP 업로드), 뒷부분(streaming
// 도우미 아래)은 로그인한 실제 모드(웹소켓 mode 'dialog')를 검사한다.
// [무엇을 지키나] 직전 마디 언어를 정확히 넘기기(서버가 애매한 마디를 번갈아 처리로 정하는 근거), 재생 중
// 듣지 않기, 방향 바꾸기의 세 결과(성공, 다시 번역 실패, 기록 삭제만 실패), 재연결 때 마지막 마디 정보 이어
// 주기, 마지막 마디를 바꿨을 때만 서버에 알리기.
// [가짜 객체] 대화 모드 테스트와 같다: deferred로 응답·재생 시점을 조종하고, 예시 모드는 입력 값을 확률로
// 쓰는 가짜 모델, 실제 모드는 정해 둔 이벤트를 돌려주는 가짜 판정기와 test/fakeSocket.ts를 쓴다.

import { afterEach, expect, it, vi } from 'vitest'
import { TranslationApi, type DialogTranslationResult, type TranslationResult } from '../translation/api'
import { SileroEndpointer, type CreateEndpointer, type LiveEndpointEvent } from '../conversation/silero'
import type { OpenMicrophone } from '../conversation/microphone'
import type { PlayTranslation } from '../conversation/player'
import { DialogSession } from './session'
import { FakeSocket } from '../test/fakeSocket'

// deferred: 밖에서 풀 수 있는 약속. utterance: 말소리 8조각 + 조용함 31조각으로 마디 하나. speechResult: 일반
// 음성 번역 결과(방향 바꾸기 응답). result: 두 사람 대화 결과, guessed면 확신 0.55로 추정 표시.
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

// 예시 모드 세션 도구. send(n): 마디 하나를 말하고 n번째 말풍선을 기다림. reply(i, body): i번째 요청에 201
// 응답.
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
  // Example mode (VITE_TRANSLATION_MOCK=true; the page still needs a login) uploads each turn over HTTP after its end.
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

// 확인: 마디는 하나씩 /api/translate/dialog로 올라가고(conversation.wav), 첫 마디는 previous_lang 없이, 둘째
// 마디는 첫 마디가 처리된 언어(ko)와 함께 간다. 멈췄다 다시 시작한 뒤 첫 마디는 다시 previous_lang 없이 간다.
// 왜: 둘째 요청이 첫 응답 전에 나가면 직전 언어를 모른 채 판별하게 된다. 요청을 순서대로 하나씩 보내는
// 이유다.
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

// 확인: 번역 음성이 실제로 나는 동안만 마이크를 쉬고 그 사이 소리는 판정기에 넣지 않으며, 끝나면 다시 듣는다.
// 멈추면 마이크를 닫고 메모리의 녹음을 지우고 재생을 취소한다.
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

// 확인: 추정으로 ko → en 처리된 마디를 바꾸면 같은 녹음을 en → ko로 /api/translate/speech에 보내고, 새 결과로
// 말풍선을 바꾼 뒤 틀린 기록(/api/history/wrong)을 DELETE하며, 새 결과를 재생한다.
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

// 확인: 반대 방향 번역이 실패하면 원래 말풍선을 그대로 두고 '원래 말풍선을 유지했습니다' 안내를 붙인다.
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

// 확인: 새 번역은 성공했고 틀린 기록 삭제만 실패하면 새 번역을 그대로 보이고, 예전 기록은 기록 화면에서 지울
// 수 있다고 알린다.
// 왜: 예전에는 삭제가 끝나야 새 번역을 보여 줘서, 삭제가 실패하면 틀린 말풍선과 기록 두 개가 남았다(5a6a17c).
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
// 실제 모드 세션 도구. liveToken은 처음 'access', 거절된 토큰을 받으면 'fresh'. 소켓은 열 때마다 sockets에
// 쌓인다.
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

// 확인: 첫 start는 mode 'dialog'만 담고(방향 없음), 마디 메시지가 순서대로 나간다. 언어 필드가 없는 final은
// 두 사람 대화 결과가 아니므로 무시하고 올바른 final만 받는다. HTTP 요청은 없다. 4401로 다시 연결하면 새
// start에 마지막으로 처리된 마디의 언어와 번호(previous_lang 'ko', previous_id 1)를 넣는다.
// 왜: 새 연결의 서버는 직전 마디를 모르므로, 넘겨주지 않으면 다음 애매한 마디를 번갈아 처리할 수 없다.
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

// 확인: 더 오래된 마디(1)를 바꿨을 때는 previous 메시지를 보내지 않고, 마지막 마디(2)를 바꿨을 때만 {type:
// 'previous', id: 2, lang: 'ko'}를 보낸다.
// 왜: 서버가 기억하는 직전 언어는 마지막 마디의 것이라, 그 마디가 바뀌었을 때만 고쳐야 다음 마디 판별이 HTTP
// 경로와 같아진다(docs/api.md).
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

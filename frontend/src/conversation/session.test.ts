// [파일 역할] 대화 모드 세션(ConversationSession) 테스트 묶음. 앞부분은 예시 모드(HTTP 업로드),
// 뒷부분(streaming 도우미 아래)은 로그인한 실제 모드(웹소켓 미리 처리)를 검사한다.
// [무엇을 지키나] 말한 순서대로 결과·재생, 요청은 한 번에 하나, 재생 중에는 듣지 않기, 한 마디 실패가 대화를
// 멈추지 않기, 멈춤·재시작 뒤 늦은 응답 무시, 재연결 때 기다리던 마디 취소. 모두 실제 사용 중 겹침·경쟁
// 상태에서 생기기 쉬운 문제들이다.
// [가짜 객체] 마이크(open)·재생(play)·요청(request)을 deferred 약속으로 만들어, 테스트가 응답·재생 끝 시점을
// 원하는 순서로 정한다. 판정기는 예시 모드에서 입력 표본 값을 그대로 확률로 쓰는 가짜 모델의
// SileroEndpointer, 실제 모드에서는 정해 둔 이벤트를 돌려주는 가짜 판정기다. 웹소켓은 test/fakeSocket.ts.

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
// deferred: 밖에서 풀거나 거절할 수 있는 약속. utterance: 말소리 8조각(0.75) + 조용함 31조각으로 마디 하나가
// 정확히 끝나는 소리. 가짜 모델이 앞 64표본 뒤 첫 값을 확률로 쓰므로 0.75는 말소리, 0은 조용함이 된다.
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
// 테스트가 만든 세션을 모두 멈춰 다음 테스트로 타이머·약속이 새지 않게 한다.
afterEach(() => { sessions.forEach((session) => session.stop()); sessions.length = 0 })

// 예시 모드 세션과 조작 도구를 만든다. send(n): 마디 하나를 말하고 화면에 n번째 마디가 생길 때까지 기다림.
// reply(i): i번째 요청에 성공 응답. capture·disconnect: 마이크 조각 넣기와 마이크 끊김.
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
  // 예시 모드(VITE_TRANSLATION_MOCK=true, 페이지는 여전히 로그인이 필요)는 마디가 끝난 뒤 클립을 HTTP로 올린다.
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

// 확인: 번역 중에도 계속 들어 둘째 마디를 받지만 요청은 하나씩 보낸다. 업로드는 conversation.wav(audio/wav,
// 머리글 44 + (말 8 + 뒤 여유 6조각) × 512표본 × 2바이트)와 방향이다. 결과는 순서대로 재생되고, 재생이
// 시작되면 마이크를 쉬게 해 그동안 들어온 소리는 판정기에 넣지 않는다. 재생이 모두 끝나면 다시 듣고 다음
// 마디를 보낸다.
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

// 확인: 번역 음성을 받는 동안에는 계속 듣다가, 실제 재생이 시작되는 순간 말하던 중인 소리를 버리고 '아직
// 끝나지 않은 말은 보내지 않았다'고 알린다.
// 왜: 재생 직전까지 듣는 시간을 늘리되, 재생 소리가 섞인 반쪽 마디를 보내지 않는다.
it('keeps listening while audio is downloading, then discards unfinished speech at playback', async () => {
  const s = setup(); await s.session.start(direction); await s.send(1); s.reply(0)
  await vi.waitFor(() => expect(s.played).toHaveLength(1))
  s.capture(new Float32Array(2 * 512).fill(.8))
  await vi.waitFor(() => expect(s.session.getSnapshot().speaking).toBe(true))
  s.played[0].begin()
  expect(s.session.getSnapshot()).toMatchObject({ playing: true, speaking: false })
  expect(s.session.getSnapshot().notice).toContain('아직 끝나지 않은 말')
})

// 확인: 첫 마디 요청이 실패하면 그 마디만 오류로 표시하고, 대기 중이던 둘째 마디를 계속 처리하며 대화는 켜진
// 채로 둔다.
it('shows a single-turn API error and continues with the queued next turn', async () => {
  const s = setup(); await s.session.start(direction); await s.send(1); await s.send(2)
  s.requests[0].reject(new Error('failed'))
  await vi.waitFor(() => expect(s.request).toHaveBeenCalledTimes(2))
  expect(s.session.getSnapshot().turns[0]).toMatchObject({ state: 'error' })
  s.reply(1)
  await vi.waitFor(() => expect(s.session.getSnapshot().turns[1].state).toBe('done'))
  expect(s.session.getSnapshot().active).toBe(true)
})

// 확인: 멈추면 진행 중인 요청의 signal이 취소되고 마이크가 닫히며 끝나지 않은 마디는 canceled가 된다. 다시
// 시작한 뒤 앞 대화의 응답이 늦게 와도 재생하지 않는다. 재생 중 멈추면 재생 signal도 취소된다.
// 왜: 대화마다 새 AbortController를 쓰는 설계가 늦은 응답을 제대로 막는지 본다.
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

// 확인: 재생이 실패하면(자동 재생 거절 등) 그 마디에 재생 오류를 표시하고 다시 듣는다. 대화를 멈춘 뒤에도
// '음성 다시 듣기'로 재생할 수 있고, 그 재생도 멈춤으로 취소된다.
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

// 확인: 판정 모델을 받는 중에 멈추면 마이크를 닫고, 늦게 도착한 판정기는 쓰지 않으며 준비 중 표시도 끈다.
// 준비 중에 들어온 소리로 요청을 보내지 않는다.
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

// 확인: 판정 모델 적재가 실패하면 오류를 보이고 마이크를 닫으며, 다시 시작하면 정상으로 시작된다(실패를
// 기억하지 않음).
it('model load failure stops capture and permits another start', async () => {
  const create = vi.fn<CreateEndpointer>().mockRejectedValueOnce(new Error('load failed'))
    .mockResolvedValue(new SileroEndpointer(async () => ({ probability: 0, h: new Float32Array(128), c: new Float32Array(128) })))
  const s = setup(create); await s.session.start(direction)
  expect(s.session.getSnapshot().error).toBeTruthy()
  expect(s.microphone.stop).toHaveBeenCalledOnce()
  await s.session.start(direction)
  expect(s.session.getSnapshot()).toMatchObject({ active: true, error: '' })
})

// 확인: 7조각(224ms)짜리 짧은 소리는 요청 없이 '짧은 소리는 건너뛰었다'고 알리고, 마이크가 끊기면 대화를
// 멈추고 연결이 끊겼다고 알린다.
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

// 로그인하면 클립은 자막 없이 동시통역 연결로 흘러가고, 서버는 192 ms 쉼에서 번역을 미리 준비한다(T77,
// docs/experiments.md 11절).
// 실제 모드 세션 도구. 판정기는 emit으로 정해 준 이벤트를 다음 push에서 그대로 돌려주는 가짜이고, 소켓은 만들
// 때마다 sockets에 쌓인다. liveToken은 처음에는 'access', 거절된 토큰을 받으면 'fresh'를 준다. start는 연결
// 열기와 ready까지 마친다.
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

// 확인: 연결이 준비되기 전에는 마이크 소리를 판정기에 넣지 않고, 첫 메시지는 토큰·방향·mode 'conversation'을
// 담은 start다. 판정 이벤트가 utterance, 소리(바이너리), pause, resume, end 메시지로 그대로 나가고, final이
// 오면 HTTP 업로드 없이 결과가 표시되고 재생된다.
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

// 확인: 서버의 final이 2번이 먼저 와도 1번이 끝나기 전에는 재생하지 않고, 1번이 오류로 끝나면 그 마디에
// 한국어 안내를 표시한 뒤 2번, 3번 순서로 재생한다.
// 왜: 미리 처리는 마디마다 끝나는 시점이 달라 결과가 뒤바뀌어 올 수 있다. 화면은 말한 순서를 지켜야 한다.
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

// 확인: 말하는 도중 재생이 시작되면 서버에 그 마디의 cancel을 보내고, 재생 뒤 그 마디의 남은 end가 와도
// 보내지 않으며 새 마디도 만들지 않는다. 재생이 끝나면 다시 듣는다.
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

// 확인: 서버가 4401로 닫으면 결과를 기다리던 마디는 canceled, 말하던 소리는 버리고 다시 말해 달라고 알리며,
// 그동안 마이크를 쉬게 한다. 거절된 토큰('access')을 넘겨 새 토큰('fresh')으로 다시 연결하고, ready 뒤 다시
// 듣는다. 마디 번호는 새 연결에서도 이어진다(3번).
// 왜: 서버는 access 토큰이 만료되면 연결을 닫는다(docs/api.md). 30분마다 오는 만료에 대화가 끊기지 않아야
// 한다.
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

// 확인: 멈추면 소켓을 닫고 기다리던 마디를 canceled로 바꾸며, 그 뒤 도착한 final은 재생하지 않는다.
it('stop closes the connection, cancels waiting turns and ignores what comes after', async () => {
  const s = streaming(); await s.start(); await s.emit(onset, audio, end)
  s.session.stop()
  expect(s.sockets[0].close).toHaveBeenCalled()
  expect(s.session.getSnapshot()).toMatchObject({ active: false, translating: false, turns: [{ id: 1, state: 'canceled' }] })
  s.sockets[0].message({ type: 'final', id: 1, result: result('1') })
  await Promise.resolve()
  expect(s.play).not.toHaveBeenCalled()
})

// 확인: 서버가 4503(모델 없음)으로 닫으면 대화를 멈추고, 안내 문구에 '동시통역'이 아니라 '대화 모드'라는 기능
// 이름이 들어간다.
it('names conversation mode when its connection fails', async () => {
  const s = streaming(); await s.start()
  s.sockets[0].closed(4503)
  expect(s.session.getSnapshot()).toMatchObject({ active: false,
    error: '대화 모드 서비스를 사용할 수 없습니다. 잠시 후 다시 시작해 주세요.' })
})

// [파일 역할] 채택된 말 끝 판정(Silero VAD v6, 쉼 1000ms, docs/experiments.md 7절)의 브라우저 구현. 16kHz
// 소리를 512표본(32ms) 조각마다 Silero 모델에 넣어 말소리 확률을 얻고, 확률 0.5 이상이 2조각 연속이면 마디를
// 시작하고, 31조각(992ms) 조용하면 마디를 끝낸다.
// [두 가지 모드] 기본 모드는 마디가 끝날 때 그 소리 전체(end 이벤트)만 내보낸다. 예시 모드의 대화 모드·두
// 사람 대화가 이 소리를 WAV로 올린다. 스트리밍 모드(createLiveEndpointer)는 마디가 진행되는 동안 소리를 audio
// 이벤트로 흘리고, 말 끝 뒤 6조각(192ms) 조용하면 pause, 그 뒤 다시 말하면 resume을 낸다. 동시통역과 로그인한
// 대화 모드·두 사람 대화가 이것으로 웹소켓에 소리를 보내, 서버가 pause에서 최종 처리를 미리
// 시작한다(docs/api.md, docs/experiments.md 11절).
// [설계 이유] 조각 크기·연속 기준·앞뒤 여유·29초 강제 끊김을 측정 도구 backend/eval/eos_eval.py와 같은 정수
// 조각 단위로 맞춰, 같은 음성에서 브라우저와 오프라인 측정이 같은 마디 경계를 낸다(f494fcc에서 확인, README
// 알려진 한계 '대화 모드'). 스트리밍 모드에서도 pause 시점에 서버가 받은 소리가 기본 모드의 WAV와 바이트
// 단위로 같게 만든다(c005648).
// [연결] conversation/session.ts, dialog/session.ts, live/session.ts가 판정기를 만들고, 모델 실행은
// sileroRuntime.ts가 맡는다.

import { SAMPLE_RATE, type EndpointEvent } from './endpointer'

// SILENCE_MS: 마디 끝으로 보는 쉼(1000ms, 측정 후 결정). SILERO_CHUNK: Silero 모델의 16kHz 입력 한
// 조각(512표본 = 32ms). CONTEXT: 앞 조각의 마지막 64표본. 조각마다 이것을 앞에 붙여 576표본을 넣는 것이
// Silero v6를 스트리밍으로 돌리는 방식이다(docs/experiments.md 7절 측정 세부).
export const SILENCE_MS = 1000
export const SILERO_CHUNK = 512
const CONTEXT = 64
// 시간을 32ms 조각 수로 반올림한다: PAUSE_FRAMES 31(992ms), PAD_FRAMES 6(192ms, 말 앞뒤 여유 200ms),
// MAX_FRAMES 906(28.992초, 앞 여유 포함 29초). 서버의 30초 한도를 넘는 마디가 나오지
// 않는다(src/conversation/assets/SOURCE.md).
// Match backend/eval/eos_eval.py group(): durations round to whole 32 ms frames.
const FRAME_MS = SILERO_CHUNK * 1000 / SAMPLE_RATE
const PAUSE_FRAMES = Math.round(SILENCE_MS / FRAME_MS)
const PAD_FRAMES = Math.round(200 / FRAME_MS)
const MAX_FRAMES = Math.round(29_000 / FRAME_MS)

// VadRunner: 모델 한 번 실행. 입력 576표본과 이전 상태(h, c 각 128개)를 받아 말소리 확률과 다음 상태를
// 돌려준다. 실제 구현은 sileroRuntime.ts이고, 테스트는 정해진 확률을 내는 가짜 함수를 넣는다.
export interface VadResult { probability: number; h: Float32Array; c: Float32Array }
export type VadRunner = (input: Float32Array, h: Float32Array, c: Float32Array) => Promise<VadResult>
// 기본 이벤트(start·end·discard)에 스트리밍 모드의 audio(보낼 소리 조각), pause(말 끝 뒤 192ms 조용함),
// resume(그 뒤 다시 말함)을 더한 것. 세 세션이 이 이벤트를 받아 처리한다.
export type LiveEndpointEvent = EndpointEvent | { type: 'audio'; samples: Float32Array } | { type: 'pause' | 'resume' }
// 세션이 쓰는 판정기 모양. push는 모델 실행이 비동기라 약속(Promise)을 돌려준다. interrupt는 진행 중인 마디를
// 버리고 그런 마디가 있었는지 알려 준다(재생을 시작할 때 세션이 '보내지 않았다' 안내를 띄우는 데 쓴다).
export interface SpeechEndpointer {
  push(samples: Float32Array): Promise<LiveEndpointEvent[]>
  interrupt(): boolean
}
export type CreateEndpointer = () => Promise<SpeechEndpointer>

// 기본 모드 판정기 만들기. sileroRuntime을 동적 import로 불러, onnxruntime-web(WASM 약 14MB, gzip 약 3.7MB,
// README)과 모델 파일을 대화 기능을 처음 시작할 때만 받는다. 번역 화면만 쓰는 사용자는 이것을 받지 않는다.
export const createSileroEndpointer: CreateEndpointer = async () => {
  // Neither ONNX Runtime nor the model is fetched on the initial page load.
  const { getSileroRunner } = await import('./sileroRuntime')
  return new SileroEndpointer(await getSileroRunner())
}

// 스트리밍 모드 판정기 만들기. 같은 모델을 쓰고 streaming=true만 다르다.
export const createLiveEndpointer: CreateEndpointer = async () => {
  const { getSileroRunner } = await import('./sileroRuntime')
  return new SileroEndpointer(await getSileroRunner(), true)
}

// Silero 말 끝 판정기.
// - tail: 앞선 push의 처리 약속. 다음 push를 그 뒤에 이어, 모델 상태(h, c)가 조각 순서대로 이어지게 한다
// - epoch: interrupt마다 올리는 번호. 그 전에 시작된 처리의 결과를 버린다
// - h·c·context: 모델의 순환 상태와 앞 조각 끝 64표본
// - pending: 512표본이 모이기 전의 나머지
// - pre: 마디 밖에서 최근 조각(최대 8개 = 앞 여유 6 + 시작을 일으킨 2)
// - frames: 진행 중인 마디의 조각들(마디 밖이면 null)
// - consecutive·speechFrames·quietFrames: 연속 말소리 수, 마디 안 말소리 조각 수, 말 뒤 연속 조용한 조각 수
export class SileroEndpointer implements SpeechEndpointer {
  private readonly run: VadRunner
  private tail: Promise<unknown> = Promise.resolve()
  private epoch = 0
  private h: Float32Array = new Float32Array(128)
  private c: Float32Array = new Float32Array(128)
  private context: Float32Array = new Float32Array(CONTEXT)
  private pending = new Float32Array(SILERO_CHUNK)
  private pendingSize = 0
  private cursor = 0
  private pre: Float32Array[] = []
  private frames: Float32Array[] | null = null
  private consecutive = 0
  private speechFrames = 0
  private quietFrames = 0
  private startSample = 0

  private readonly streaming: boolean
  constructor(run: VadRunner, streaming = false) { this.run = run; this.streaming = streaming }

  // 재생 시작·멈춤·재연결 때 부른다. 모델 상태와 버퍼를 처음으로 되돌리고 epoch를 올려, 이미 실행 중인 모델
  // 호출의 결과가 돌아와도 쓰지 않게 한다. 진행 중인 마디가 있었으면 true.
  interrupt(): boolean {
    const discarded = this.frames !== null
    this.epoch++
    this.h = new Float32Array(128); this.c = new Float32Array(128)
    this.context = new Float32Array(CONTEXT)
    this.pendingSize = 0; this.pre = []; this.frames = null
    this.consecutive = this.speechFrames = this.quietFrames = 0
    return discarded
  }

  // 소리 조각을 넣고, 그 조각까지 처리해서 생긴 이벤트를 약속으로 돌려준다. 호출 쪽이 버퍼를 다시 써도
  // 안전하도록 복사해 두고, 처리를 tail 뒤에 이어 순서를 지킨다. tail에는 오류를 삼킨 약속을 둬 한 번의
  // 실패가 뒤의 처리를 막지 않게 하고, 실패 자체는 돌려준 약속으로 세션에 전한다.
  push(samples: Float32Array): Promise<LiveEndpointEvent[]> {
    const epoch = this.epoch
    const owned = samples.slice()
    const task = this.tail.then(() => this.process(owned, epoch))
    this.tail = task.catch(() => undefined)
    return task
  }

  private async process(samples: Float32Array, epoch: number): Promise<LiveEndpointEvent[]> {
    if (epoch !== this.epoch) return []
    const events: LiveEndpointEvent[] = []
    for (const sample of samples) {
      this.pending[this.pendingSize++] = sample
      if (this.pendingSize !== SILERO_CHUNK) continue
      const frame = this.pending
      this.pending = new Float32Array(SILERO_CHUNK); this.pendingSize = 0
      // 모델 입력 = 앞 조각 끝 64표본 + 이번 512표본. 결과로 받은 상태와 이번 조각의 끝 64표본을 다음 조각에
      // 넘긴다. 확률이 숫자가 아니면 모델 오류로 보고 멈춘다.
      const input = new Float32Array(CONTEXT + SILERO_CHUNK)
      input.set(this.context); input.set(frame, CONTEXT)
      const result = await this.run(input, this.h, this.c)
      // Stop/playback can interrupt while session.run is pending. Never restore its old state.
      if (epoch !== this.epoch) return []
      if (!Number.isFinite(result.probability)) throw new Error('Invalid VAD output')
      this.h = result.h; this.c = result.c; this.context = frame.slice(-CONTEXT)
      this.cursor += SILERO_CHUNK
      // 말소리 판정 기준: 확률 0.5 이상(docs/experiments.md 7절 후보 S).
      const speech = result.probability >= 0.5
      // 마디 밖: 최근 조각을 pre에 8개까지 모으고, 말소리 2조각(64ms) 연속이면 마디를 시작한다. 시작한 마디는
      // pre 전체, 즉 말 앞 192ms + 시작 2조각으로 시작한다. 스트리밍 모드면 이 조각들을 바로 audio로
      // 보낸다(docs/api.md: 말 앞 192ms부터).
      if (!this.frames) {
        this.consecutive = speech ? this.consecutive + 1 : 0
        this.pre.push(frame)
        if (this.pre.length > PAD_FRAMES + 2) this.pre.shift()
        if (this.consecutive < 2) continue
        this.frames = [...this.pre]
        this.startSample = this.cursor - this.frames.length * SILERO_CHUNK
        this.speechFrames = 2; this.quietFrames = 0
        events.push({ type: 'start', startSample: this.startSample })
        if (this.streaming) for (const samples of this.frames) events.push({ type: 'audio', samples })
      } else {
        // 마디 안: 조각을 모으고 말소리·조용함을 센다.
        // 스트리밍 규칙: 말 뒤 조용한 조각은 6개(192ms)까지만 보내고, 6개째에 pause를 보낸다. 7개째부터는
        // 보내지 않고 들고 있는다. 그래서 pause 때 서버가 가진 소리는 마디가 여기서 끝났을 때의 WAV(뒤 여유
        // 6조각)와 정확히 같다. 6개 이상 쉰 뒤 다시 말하면 resume을 보내고, 들고 있던 조용한 조각과 이번
        // 조각을 보내 마디 안의 쉼이 사라지지 않게 한다.
        const previousQuiet = this.quietFrames
        this.frames.push(frame)
        if (speech) { this.speechFrames++; this.quietFrames = 0 } else this.quietFrames++
        if (this.streaming) {
          if (speech && previousQuiet >= PAD_FRAMES) {
            events.push({ type: 'resume' })
            // Keep the silence after the tail padding locally until speech resumes. The pause
            // snapshot must equal the conversation WAV, but an internal gap must not disappear.
            for (const samples of this.frames.slice(-(previousQuiet - PAD_FRAMES + 1))) {
              events.push({ type: 'audio', samples })
            }
          } else if (this.quietFrames <= PAD_FRAMES) events.push({ type: 'audio', samples: frame })
          if (this.quietFrames === PAD_FRAMES) events.push({ type: 'pause' })
        }
      }
      // 끝내기: 31조각 조용하거나 29초에 닿으면 마디를 닫는다. 뒤의 조용한 조각은 6개만 남기고, 말소리 조각
      // 합이 250ms(8조각 = 256ms) 미만이면 기침·짧은 잡음으로 보고 버린다(discard).
      const forced = this.frames.length >= MAX_FRAMES
      if (!forced && this.quietFrames < PAUSE_FRAMES) continue
      const keep = this.frames.length - Math.max(0, this.quietFrames - PAD_FRAMES)
      if (this.speechFrames * FRAME_MS >= 250) {
        const clip = new Float32Array(keep * SILERO_CHUNK)
        for (let i = 0; i < keep; i++) clip.set(this.frames[i], i * SILERO_CHUNK)
        events.push({ type: 'end', samples: clip, startSample: this.startSample,
          endSample: this.startSample + clip.length, detectedAtSample: this.cursor, forced })
      } else events.push({ type: 'discard' })
      // Pre-roll only from frames this clip left out, at most six: after a normal end that is the silence
      // past the tail padding; a forced split in speech leaves none, so the next clip never repeats audio.
      // A forced split in a long pause drops the quiet frames before those six, as a normal end does.
      this.pre = this.frames.slice(keep).slice(-PAD_FRAMES)
      this.frames = null; this.consecutive = this.speechFrames = this.quietFrames = 0
    }
    return events
  }
}

// Candidate E, T33. Keep these calculations in sync with the offline Python measurement.
// [파일 역할] 대화 모드의 말 끝 판정 후보 E(음량 기준)를 브라우저에서 계산하는 구현. docs/experiments.md
// 7절에서 후보 S(Silero VAD)와 비교했고, 잡음 속에서 문장을 놓치고 일찍 끊는 문제로 S가 채택됐다. 그래서 지금
// 화면은 이 파일에서 공용 상수 SAMPLE_RATE와 이벤트 타입 EndpointEvent만 가져다 쓰고(pcm.ts, silero.ts),
// Endpointer 클래스는 화면에서 쓰지 않는다.
// [설계 이유] 첫 줄 주석대로 계산을 측정 도구(backend/eval/eos_eval.py의 E)와 똑같이 두어, 오프라인 측정
// 결과가 이 브라우저 코드의 동작을 그대로 대표하게 했다. 계산 규칙: 20ms 프레임 dB = 20·log10(RMS + 1e-10),
// 처음 25프레임으로 잡음 수준을 잡고, 잡음보다 10dB 넘게 크면 말소리, 3프레임 연속이면
// 시작(docs/experiments.md 7절 측정 세부).

// SAMPLE_RATE: 앱 전체가 쓰는 소리 표본화율(16kHz). 서버 음성 인식과 Silero VAD의 입력 규격이다.
// FRAME_SAMPLES: 20ms 프레임(16000 × 0.02 = 320표본). PAUSE_CANDIDATES: 측정에서 비교한 쉼 기준
// 후보(500·700·1000ms).
export const SAMPLE_RATE = 16_000
export const FRAME_SAMPLES = 320
export const PAUSE_CANDIDATES = [500, 700, 1000] as const
export type PauseMs = typeof PAUSE_CANDIDATES[number]
// 판정이 내보내는 이벤트. start: 마디 시작(앞 여유 소리 포함 시작 위치, 표본 단위). end: 마디 끝(잘라 낸
// 소리, 시작·끝 위치, 판정한 시각, 29초 강제 끊김 여부). discard: 말소리가 너무 짧아 버림. Silero 구현도 같은
// 이벤트를 쓴다(silero.ts의 LiveEndpointEvent가 이것을 넓힌다).
export type EndpointEvent =
  | { type: 'start'; startSample: number }
  | { type: 'end'; startSample: number; endSample: number; detectedAtSample: number; samples: Float32Array; forced: boolean }
  | { type: 'discard' }

// 프레임의 음량(dBFS). RMS에 1e-10을 더해 완전한 무음에서 log10(0) = −무한대가 되지 않게 한다.
export function frameDb(frame: Float32Array): number {
  let squares = 0
  for (const sample of frame) squares += sample * sample
  return 20 * Math.log10(Math.sqrt(squares / frame.length) + 1e-10)
}

// 음량 기준 말 끝 판정기. 소리 조각을 아무 길이로 받아 20ms 프레임으로 모으고 프레임마다 판정한다.
// - calibration·noise: 잡음 수준(dB). 처음 25프레임(0.5초)의 중앙값으로 정하고, 이후 말소리가 아닌 프레임으로
//   천천히 따라간다
// - pre: 마디 시작 전 최근 프레임(최대 13개). 말 앞부분이 잘리지 않게 시작 때 앞에 붙인다
// - frames: 진행 중인 마디의 프레임(마디 밖이면 null)
// - consecutive·speechFrames·quietFrames: 연속 말소리 수, 마디 안 말소리 수, 말 뒤 연속 조용한 수
// - cursor·startSample: 지금까지 처리한 표본 수와 마디 시작 위치
export class Endpointer {
  private readonly pauseFrames: number
  private calibration: number[] = []
  private noise: number | undefined
  private pending = new Float32Array(FRAME_SAMPLES)
  private pendingSize = 0
  private cursor = 0
  private pre: Float32Array[] = []
  private frames: Float32Array[] | null = null
  private consecutive = 0
  private speechFrames = 0
  private quietFrames = 0
  private startSample = 0

  // 쉼 기준(ms)을 20ms 프레임 수로 바꿔 둔다.
  constructor(pauseMs: PauseMs = 700) { this.pauseFrames = pauseMs / 20 }

  // Playback creates a discontinuity: discard unfinished speech/pre-roll, preserve the noise estimate.
  interrupt(): boolean {
    const discarded = this.frames !== null
    this.pendingSize = 0
    this.pre = []
    this.frames = null
    this.consecutive = this.speechFrames = this.quietFrames = 0
    return discarded
  }

  // 소리를 넣고 그 사이 생긴 이벤트들을 돌려준다. 프레임 경계에 걸친 나머지 표본은 pending에 남겨 다음 호출로
  // 넘긴다.
  push(samples: Float32Array): EndpointEvent[] {
    const events: EndpointEvent[] = []
    for (const sample of samples) {
      this.pending[this.pendingSize++] = sample
      if (this.pendingSize !== FRAME_SAMPLES) continue
      const frame = this.pending
      this.pending = new Float32Array(FRAME_SAMPLES)
      this.pendingSize = 0
      this.cursor += FRAME_SAMPLES
      const db = frameDb(frame)
      // The first 25 frames only calibrate. Median is the 13th sorted dB value.
      if (this.noise === undefined) {
        this.calibration.push(db)
        this.remember(frame)
        if (this.calibration.length === 25) this.noise = [...this.calibration].sort((a, b) => a - b)[12]
        continue
      }
      // 잡음보다 10dB 넘게 크면 말소리. 말소리가 아닌 프레임으로 잡음 수준을 지수 이동 평균(새 값 5%)으로
      // 갱신해, 주변 소음이 바뀌어도 기준이 천천히 따라가게 한다.
      const speech = db > this.noise + 10
      if (!speech) this.noise = 0.95 * this.noise + 0.05 * db

      // 마디 밖: 말소리가 3프레임(60ms) 연속이면 마디를 시작한다. 짧은 딸깍 소리 하나로 시작하지 않게 하려는
      // 것이다.
      if (!this.frames) {
        this.consecutive = speech ? this.consecutive + 1 : 0
        this.remember(frame)
        if (this.consecutive < 3) continue
        // Last 13 frames = 10 before the first speech frame, plus the 3 triggering frames.
        this.frames = [...this.pre]
        this.startSample = this.cursor - this.frames.length * FRAME_SAMPLES
        this.speechFrames = 3
        this.quietFrames = 0
        events.push({ type: 'start', startSample: this.startSample })
      } else {
        // 마디 안: 프레임을 모으며 말소리·조용함을 센다.
        this.frames.push(frame)
        if (speech) { this.speechFrames++; this.quietFrames = 0 }
        else this.quietFrames++
      }

      // 29 seconds includes pre-roll; no uploaded clip can cross this limit.
      const forced = this.frames.length >= 29_000 / 20
      // 쉼 기준만큼 조용하거나 29초에 닿으면 마디를 끝낸다. 끝의 조용한 프레임은 10개(200ms)만 남기고 잘라
      // 내고, 말소리가 250ms 미만이면 기침·짧은 잡음으로 보고 버린다(docs/experiments.md 7절 공통 규칙).
      if (!forced && this.quietFrames < this.pauseFrames) continue
      const keep = this.frames.length - Math.max(0, this.quietFrames - 10)
      if (this.speechFrames * 20 >= 250) {
        const clip = new Float32Array(keep * FRAME_SAMPLES)
        for (let i = 0; i < keep; i++) clip.set(this.frames[i], i * FRAME_SAMPLES)
        events.push({ type: 'end', samples: clip, startSample: this.startSample,
          endSample: this.startSample + clip.length, detectedAtSample: this.cursor, forced })
      } else events.push({ type: 'discard' })
      // At a forced split do not repeat old speech as pre-roll in the next clip.
      this.pre = forced ? [] : this.frames.slice(-10)
      this.frames = null
      this.consecutive = this.speechFrames = this.quietFrames = 0
    }
    return events
  }

  private remember(frame: Float32Array) {
    this.pre.push(frame)
    if (this.pre.length > 13) this.pre.shift()
  }
}

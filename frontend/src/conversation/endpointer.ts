// Candidate E, T33. Keep these calculations in sync with the offline Python measurement.
export const SAMPLE_RATE = 16_000
export const FRAME_SAMPLES = 320
export const PAUSE_CANDIDATES = [500, 700, 1000] as const
export type PauseMs = typeof PAUSE_CANDIDATES[number]
export type EndpointEvent =
  | { type: 'start'; startSample: number }
  | { type: 'end'; startSample: number; endSample: number; detectedAtSample: number; samples: Float32Array; forced: boolean }
  | { type: 'discard' }

export function frameDb(frame: Float32Array): number {
  let squares = 0
  for (const sample of frame) squares += sample * sample
  return 20 * Math.log10(Math.sqrt(squares / frame.length) + 1e-10)
}

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
      const speech = db > this.noise + 10
      if (!speech) this.noise = 0.95 * this.noise + 0.05 * db

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
        this.frames.push(frame)
        if (speech) { this.speechFrames++; this.quietFrames = 0 }
        else this.quietFrames++
      }

      // 29 seconds includes pre-roll; no uploaded clip can cross this limit.
      const forced = this.frames.length >= 29_000 / 20
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

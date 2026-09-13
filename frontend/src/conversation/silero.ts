import { SAMPLE_RATE, type EndpointEvent } from './endpointer'

export const SILENCE_MS = 1000
export const SILERO_CHUNK = 512
const CONTEXT = 64
// Match backend/eval/eos_eval.py group(): durations round to whole 32 ms frames.
const FRAME_MS = SILERO_CHUNK * 1000 / SAMPLE_RATE
const PAUSE_FRAMES = Math.round(SILENCE_MS / FRAME_MS)
const PAD_FRAMES = Math.round(200 / FRAME_MS)
const MAX_FRAMES = Math.round(29_000 / FRAME_MS)

export interface VadResult { probability: number; h: Float32Array; c: Float32Array }
export type VadRunner = (input: Float32Array, h: Float32Array, c: Float32Array) => Promise<VadResult>
export interface SpeechEndpointer {
  push(samples: Float32Array): Promise<EndpointEvent[]>
  interrupt(): boolean
}
export type CreateEndpointer = () => Promise<SpeechEndpointer>

export const createSileroEndpointer: CreateEndpointer = async () => {
  // Neither ONNX Runtime nor the model is fetched on the initial page load.
  const { getSileroRunner } = await import('./sileroRuntime')
  return new SileroEndpointer(await getSileroRunner())
}

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

  constructor(run: VadRunner) { this.run = run }

  interrupt(): boolean {
    const discarded = this.frames !== null
    this.epoch++
    this.h = new Float32Array(128); this.c = new Float32Array(128)
    this.context = new Float32Array(CONTEXT)
    this.pendingSize = 0; this.pre = []; this.frames = null
    this.consecutive = this.speechFrames = this.quietFrames = 0
    return discarded
  }

  push(samples: Float32Array): Promise<EndpointEvent[]> {
    const epoch = this.epoch
    const owned = samples.slice()
    const task = this.tail.then(() => this.process(owned, epoch))
    this.tail = task.catch(() => undefined)
    return task
  }

  private async process(samples: Float32Array, epoch: number): Promise<EndpointEvent[]> {
    if (epoch !== this.epoch) return []
    const events: EndpointEvent[] = []
    for (const sample of samples) {
      this.pending[this.pendingSize++] = sample
      if (this.pendingSize !== SILERO_CHUNK) continue
      const frame = this.pending
      this.pending = new Float32Array(SILERO_CHUNK); this.pendingSize = 0
      const input = new Float32Array(CONTEXT + SILERO_CHUNK)
      input.set(this.context); input.set(frame, CONTEXT)
      const result = await this.run(input, this.h, this.c)
      // Stop/playback can interrupt while session.run is pending. Never restore its old state.
      if (epoch !== this.epoch) return []
      if (!Number.isFinite(result.probability)) throw new Error('Invalid VAD output')
      this.h = result.h; this.c = result.c; this.context = frame.slice(-CONTEXT)
      this.cursor += SILERO_CHUNK
      const speech = result.probability >= 0.5
      if (!this.frames) {
        this.consecutive = speech ? this.consecutive + 1 : 0
        this.pre.push(frame)
        if (this.pre.length > PAD_FRAMES + 2) this.pre.shift()
        if (this.consecutive < 2) continue
        this.frames = [...this.pre]
        this.startSample = this.cursor - this.frames.length * SILERO_CHUNK
        this.speechFrames = 2; this.quietFrames = 0
        events.push({ type: 'start', startSample: this.startSample })
      } else {
        this.frames.push(frame)
        if (speech) { this.speechFrames++; this.quietFrames = 0 } else this.quietFrames++
      }
      const forced = this.frames.length >= MAX_FRAMES
      if (!forced && this.quietFrames < PAUSE_FRAMES) continue
      const keep = this.frames.length - Math.max(0, this.quietFrames - PAD_FRAMES)
      if (this.speechFrames * FRAME_MS >= 250) {
        const clip = new Float32Array(keep * SILERO_CHUNK)
        for (let i = 0; i < keep; i++) clip.set(this.frames[i], i * SILERO_CHUNK)
        events.push({ type: 'end', samples: clip, startSample: this.startSample,
          endSample: this.startSample + clip.length, detectedAtSample: this.cursor, forced })
      } else events.push({ type: 'discard' })
      this.pre = this.frames.slice(-PAD_FRAMES)
      this.frames = null; this.consecutive = this.speechFrames = this.quietFrames = 0
    }
    return events
  }
}

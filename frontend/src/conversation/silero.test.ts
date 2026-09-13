import { beforeEach, expect, it, vi } from 'vitest'
import { SileroEndpointer, SILERO_CHUNK, createSileroEndpointer } from './silero'
import type { EndpointEvent } from './endpointer'

const ort = vi.hoisted(() => ({ run: vi.fn(), create: vi.fn(), env: { wasm: {} as Record<string, unknown> } }))
vi.mock('onnxruntime-web/wasm', () => ({
  env: ort.env, InferenceSession: { create: ort.create },
  Tensor: class {
    type: string; data: Float32Array; dims: number[]
    constructor(type: string, data: Float32Array, dims: number[]) { this.type = type; this.data = data; this.dims = dims }
  },
}))
const samples = (frames: number, value = 0) => new Float32Array(frames * SILERO_CHUNK).fill(value)
const ends = (events: EndpointEvent[]) => events.filter((e) => e.type === 'end')

beforeEach(() => {
  vi.resetModules(); ort.run.mockReset(); ort.create.mockReset()
  ort.create.mockResolvedValue({ run: ort.run })
})

async function detector(probabilities: number[]) {
  let index = 0
  ort.run.mockImplementation(async () => {
    const probability = probabilities[index++] ?? 0
    return { speech_probs: { data: [probability] },
      hn: { data: new Float32Array(128).fill(index) }, cn: { data: new Float32Array(128).fill(-index) } }
  })
  return createSileroEndpointer()
}

it('loads one session lazily, pins WASM to local assets and reuses it across starts', async () => {
  expect(ort.create).not.toHaveBeenCalled()
  const { getSileroRunner } = await import('./sileroRuntime')
  await Promise.all([getSileroRunner(), getSileroRunner()])
  expect(ort.create).toHaveBeenCalledTimes(1)
  expect(ort.create.mock.calls[0][0]).toContain('silero_vad_v6.onnx')
  expect(ort.create.mock.calls[0][1]).toEqual({ executionProviders: ['wasm'] })
  expect(ort.env.wasm.numThreads).toBe(1)
  const paths = ort.env.wasm.wasmPaths as Record<string, string>
  expect(paths.wasm).toContain('ort-wasm-simd-threaded.wasm')
  expect(paths.mjs).toContain('ort-wasm-simd-threaded.mjs')
  expect(Object.values(paths).every((path) => !/^https?:/.test(path))).toBe(true)
})

it('retries a failed model load instead of caching the rejection', async () => {
  ort.create.mockRejectedValueOnce(new Error('missing file'))
  const { getSileroRunner } = await import('./sileroRuntime')
  await expect(getSileroRunner()).rejects.toThrow('missing file')
  await getSileroRunner()
  expect(ort.create).toHaveBeenCalledTimes(2)
})

it('uses [1,576] input, 64 previous samples and carries separate [1,1,128] h/c tensors', async () => {
  const vad = await detector([0, 0])
  await vad.push(samples(1, .25).slice(0, 400))
  expect(ort.run).not.toHaveBeenCalled()
  await vad.push(new Float32Array(112).fill(.25))
  await vad.push(samples(1, .75))
  const [a, b] = ort.run.mock.calls.map(([feed]) => feed)
  expect(a.input.dims).toEqual([1, 576]); expect(a.input.type).toBe('float32')
  expect([...a.input.data.slice(0, 64)]).toEqual([...new Float32Array(64)])
  expect([...b.input.data.slice(0, 64)]).toEqual([...new Float32Array(64).fill(.25)])
  expect([...b.input.data.slice(64)]).toEqual([...samples(1, .75)])
  expect(a.h.dims).toEqual([1, 1, 128]); expect(a.c.dims).toEqual([1, 1, 128])
  expect(a.h.data.every((n: number) => n === 0)).toBe(true)
  expect(b.h.data.every((n: number) => n === 1)).toBe(true)
  expect(b.c.data.every((n: number) => n === -1)).toBe(true)
})

it('requires two consecutive >=0.5 frames and rejects scattered noise', async () => {
  const vad = await detector([.5, .49, .7, .1, .5, .5])
  expect(await vad.push(samples(5))).toEqual([])
  expect(await vad.push(samples(1))).toEqual([{ type: 'start', startSample: 0 }])
})

it('ends at the measured 31 quiet frames, retains six pre/post frames, with exact sample contents', async () => {
  const vad = await detector([...Array(10).fill(0), ...Array(8).fill(.5), ...Array(31).fill(0)])
  const pcm = Float32Array.from({ length: 49 * 512 }, (_, i) => i / (49 * 512))
  const initial = await vad.push(pcm.slice(0, 48 * 512))
  expect(initial).toEqual([{ type: 'start', startSample: 4 * 512 }])
  const [end] = ends(await vad.push(pcm.slice(48 * 512)))
  expect(end).toMatchObject({ startSample: 4 * 512, endSample: 24 * 512, detectedAtSample: 49 * 512, forced: false })
  expect(end.samples).toEqual(pcm.slice(4 * 512, 24 * 512))
})

it('discards less than 250ms of speech (7 frames) but accepts 8 frames', async () => {
  const vad = await detector([...Array(7).fill(1), ...Array(31).fill(0), ...Array(8).fill(1), ...Array(31).fill(0)])
  expect(await vad.push(samples(38))).toEqual([{ type: 'start', startSample: 0 }, { type: 'discard' }])
  expect(ends(await vad.push(samples(39)))).toHaveLength(1)
})

it('counts total speech across short internal pauses and restarts the silence counter', async () => {
  const vad = await detector([...Array(4).fill(1), ...Array(30).fill(0), ...Array(4).fill(1), ...Array(31).fill(0)])
  const result = ends(await vad.push(samples(69)))
  expect(result).toHaveLength(1)
  expect(result[0].detectedAtSample).toBe(69 * 512)
})

it('forces at 906 frames including pre-roll (<29s), then continues listening', async () => {
  const vad = await detector([...Array(10).fill(0), ...Array(920).fill(1), ...Array(31).fill(0)])
  const result = ends(await vad.push(samples(961)))
  expect(result).toHaveLength(2)
  expect(result[0]).toMatchObject({ forced: true, startSample: 4 * 512, detectedAtSample: 910 * 512 })
  expect(result[0].samples.length).toBe(906 * 512)
  expect(result[0].samples.length / 16_000).toBeLessThanOrEqual(29)
  expect(result[1].forced).toBe(false)
})

it('serializes concurrent chunks rather than running frames with stale state', async () => {
  const vad = await detector([1, 1])
  const events = await Promise.all([vad.push(samples(1)), vad.push(samples(1))])
  expect(events[1][0].type).toBe('start')
  expect(ort.run.mock.calls[1][0].h.data[0]).toBe(1)
})

it('invalidates pending runs and queued chunks at playback, resets context and h/c', async () => {
  let release!: (value: { probability: number; h: Float32Array; c: Float32Array }) => void
  const run = vi.fn().mockImplementationOnce(() => new Promise((resolve) => { release = resolve }))
    .mockResolvedValue({ probability: 0, h: new Float32Array(128), c: new Float32Array(128) })
  const vad = new SileroEndpointer(run)
  const old = vad.push(samples(1, .75))
  const queued = vad.push(samples(1, .75))
  await vi.waitFor(() => expect(run).toHaveBeenCalledOnce())
  vad.interrupt()
  release({ probability: 1, h: new Float32Array(128).fill(9), c: new Float32Array(128).fill(9) })
  expect(await old).toEqual([]); expect(await queued).toEqual([])
  await vad.push(samples(1, .25))
  expect(run).toHaveBeenCalledTimes(2)
  expect(run.mock.calls[1][0].slice(0, 64).every((n: number) => n === 0)).toBe(true)
  expect(run.mock.calls[1][1].every((n: number) => n === 0)).toBe(true)
})

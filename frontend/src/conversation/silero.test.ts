// [파일 역할] Silero 말 끝 판정기(silero.ts)와 모델 로더(sileroRuntime.ts)의 기본 모드 테스트 묶음. 모델 로딩
// 방식, 모델 입력 모양, 시작·끝·버림 규칙, 29초 강제 끊김 뒤 이어지는 마디, 비동기 실행 순서와 중단을
// 확인한다.
// [왜 중요한가] 이 규칙들은 측정 도구(backend/eval/eos_eval.py)로 고른 설정과 같아야 오프라인 측정 결과(잘림
// 2.2~3.1%, 판정 지연 약 1.17초, README)가 실제 화면에도 성립한다. 숫자 하나가 바뀌어도 테스트가 잡도록 조각
// 수·표본 위치를 정확히 비교한다.
// [가짜 객체] onnxruntime-web/wasm 모듈 전체를 vi.mock으로 바꿔 실제 WASM 없이 돈다. vi.hoisted는 mock 함수가
// import보다 먼저 만들어지게 한다.

import { beforeEach, expect, it, vi } from 'vitest'
import { SileroEndpointer, SILERO_CHUNK, createSileroEndpointer } from './silero'
import type { LiveEndpointEvent } from './silero'

const ort = vi.hoisted(() => ({ run: vi.fn(), create: vi.fn(), env: { wasm: {} as Record<string, unknown> } }))
vi.mock('onnxruntime-web/wasm', () => ({
  env: ort.env, InferenceSession: { create: ort.create },
  Tensor: class {
    type: string; data: Float32Array; dims: number[]
    constructor(type: string, data: Float32Array, dims: number[]) { this.type = type; this.data = data; this.dims = dims }
  },
}))
// samples: 32ms 조각 n개 길이의 소리. ends: 이벤트 중 end만 고르기.
const samples = (frames: number, value = 0) => new Float32Array(frames * SILERO_CHUNK).fill(value)
const ends = (events: LiveEndpointEvent[]) => events.filter((e) => e.type === 'end')

// resetModules로 sileroRuntime을 테스트마다 새로 불러, 모듈 안에 기억된 세션(loading)이 다른 테스트로 새지
// 않게 한다.
beforeEach(() => {
  vi.resetModules(); ort.run.mockReset(); ort.create.mockReset()
  ort.create.mockResolvedValue({ run: ort.run })
})

// 정해진 확률 순서를 내는 가짜 모델로 판정기를 만든다. 다음 상태 h는 호출 번호, c는 그 음수로 채워, 상태가
// 다음 호출로 제대로 이어지는지 확인할 수 있게 한다.
async function detector(probabilities: number[]) {
  let index = 0
  ort.run.mockImplementation(async () => {
    const probability = probabilities[index++] ?? 0
    return { speech_probs: { data: [probability] },
      hn: { data: new Float32Array(128).fill(index) }, cn: { data: new Float32Array(128).fill(-index) } }
  })
  return createSileroEndpointer()
}

// 확인: 불러오기 전에는 세션을 만들지 않고, 동시에 두 번 불러도 세션은 하나이며, 로컬 모델 파일과 WASM 실행
// 제공자, 스레드 1개, http(s)가 아닌 로컬 WASM 주소를 쓴다.
// 왜: 대화 기능을 쓰지 않는 사용자에게 큰 파일을 받게 하지 않고, 외부 CDN 없이 배포본 안의 자산만 쓰는 약속을
// 지킨다.
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

// 확인: 모델 적재가 한 번 실패해도 그 실패를 기억하지 않고 다음 호출에서 다시 만든다. 네트워크가 잠깐 끊겼을
// 때 페이지를 새로 고치지 않아도 다시 시작할 수 있다.
it('retries a failed model load instead of caching the rejection', async () => {
  ort.create.mockRejectedValueOnce(new Error('missing file'))
  const { getSileroRunner } = await import('./sileroRuntime')
  await expect(getSileroRunner()).rejects.toThrow('missing file')
  await getSileroRunner()
  expect(ort.create).toHaveBeenCalledTimes(2)
})

// 확인: 512표본이 다 모여야 모델을 부르고(400 + 112), 입력은 [1, 576]에 앞 조각 끝 64표본(첫 조각은 0)을
// 붙이며, h·c는 [1, 1, 128] 따로이고 앞 호출의 결과 상태가 다음 호출로 넘어간다.
// 왜: Silero v6의 스트리밍 입력 규격이다. 이것이 틀리면 확률이 오프라인 측정과 달라진다.
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

// 확인: 0.5 이상이 띄엄띄엄 나오면 시작하지 않고, 0.5가 두 조각 연속일 때 처음 시작한다(경계값 0.5 포함,
// 0.49는 아님).
it('requires two consecutive >=0.5 frames and rejects scattered noise', async () => {
  const vad = await detector([.5, .49, .7, .1, .5, .5])
  expect(await vad.push(samples(5))).toEqual([])
  expect(await vad.push(samples(1))).toEqual([{ type: 'start', startSample: 0 }])
})

// 확인: 31조각 조용하면 끝나고, 마디는 말 앞 6조각부터 말 뒤 6조각까지(4~24번 조각)이며 그 안의 표본 값이
// 입력과 정확히 같다. 판정 시각은 마지막 조용한 조각 끝(49조각)이다.
it('ends at the measured 31 quiet frames, retains six pre/post frames, with exact sample contents', async () => {
  const vad = await detector([...Array(10).fill(0), ...Array(8).fill(.5), ...Array(31).fill(0)])
  const pcm = Float32Array.from({ length: 49 * 512 }, (_, i) => i / (49 * 512))
  const initial = await vad.push(pcm.slice(0, 48 * 512))
  expect(initial).toEqual([{ type: 'start', startSample: 4 * 512 }])
  const [end] = ends(await vad.push(pcm.slice(48 * 512)))
  expect(end).toMatchObject({ startSample: 4 * 512, endSample: 24 * 512, detectedAtSample: 49 * 512, forced: false })
  expect(end.samples).toEqual(pcm.slice(4 * 512, 24 * 512))
})

// 확인: 말소리 7조각(224ms)은 버리고 8조각(256ms)은 마디로 보낸다. 기침·짧은 잡음을 거르는 250ms 규칙의
// 경계다.
it('discards less than 250ms of speech (7 frames) but accepts 8 frames', async () => {
  const vad = await detector([...Array(7).fill(1), ...Array(31).fill(0), ...Array(8).fill(1), ...Array(31).fill(0)])
  expect(await vad.push(samples(38))).toEqual([{ type: 'start', startSample: 0 }, { type: 'discard' }])
  expect(ends(await vad.push(samples(39)))).toHaveLength(1)
})

// 확인: 30조각(1초 미만) 쉬고 다시 말하면 같은 마디로 이어지고 조용함 계산이 처음부터 다시 시작된다. 말소리 4
// + 4조각이 합쳐져 250ms를 넘으므로 버려지지 않는다.
it('counts total speech across short internal pauses and restarts the silence counter', async () => {
  const vad = await detector([...Array(4).fill(1), ...Array(30).fill(0), ...Array(4).fill(1), ...Array(31).fill(0)])
  const result = ends(await vad.push(samples(69)))
  expect(result).toHaveLength(1)
  expect(result[0].detectedAtSample).toBe(69 * 512)
})

// 확인: 계속 말하면 앞 여유 포함 906조각(28.992초)에서 강제로 끊고, 그 뒤에도 계속 들어 다음 마디를 만든다.
// 서버의 30초 한도를 넘는 마디를 보내지 않는다.
it('forces at 906 frames including pre-roll (<29s), then continues listening', async () => {
  const vad = await detector([...Array(10).fill(0), ...Array(920).fill(1), ...Array(31).fill(0)])
  const result = ends(await vad.push(samples(961)))
  expect(result).toHaveLength(2)
  expect(result[0]).toMatchObject({ forced: true, startSample: 4 * 512, detectedAtSample: 910 * 512 })
  expect(result[0].samples.length).toBe(906 * 512)
  expect(result[0].samples.length / 16_000).toBeLessThanOrEqual(29)
  expect(result[1].forced).toBe(false)
})

// 확인: 말하는 도중 강제로 끊긴 뒤 다음 마디는 앞 마디가 끝난 표본에서 바로 시작하고 소리를 겹쳐 보내지
// 않는다.
// 왜: 예전에는 끝난 마디의 마지막 6조각을 다음 마디 앞에 다시 붙여, 그 사이 단어가 두 번 인식될 수
// 있었다(11f58a0).
it('starts the next clip where a forced split in speech stopped, repeating none of its audio', async () => {
  const vad = await detector([...Array(10).fill(0), ...Array(920).fill(1), ...Array(31).fill(0)])
  const pcm = Float32Array.from({ length: 961 * 512 }, (_, i) => i / (961 * 512))
  const [first, second] = ends(await vad.push(pcm))
  expect(first).toMatchObject({ forced: true, startSample: 4 * 512, endSample: 910 * 512 })
  expect(second).toMatchObject({ forced: false, startSample: 910 * 512, endSample: 936 * 512 })
  expect(first.samples.at(-1)).toBe(pcm[910 * 512 - 1])
  expect(second.samples).toEqual(pcm.slice(910 * 512, 936 * 512))
})

// 확인: 짧은 쉼(9조각) 중에 강제로 끊기면, 앞 마디가 남긴 3조각만 다음 마디의 앞 여유가 되어 두 마디가
// 빈틈없이 이어진다.
it('uses only the frames a forced split in a pause left out as the next pre-roll', async () => {
  // Nine quiet frames when the limit hits: the forced clip keeps six and leaves out three.
  const vad = await detector([...Array(10).fill(0), ...Array(891).fill(1), ...Array(9).fill(0),
    ...Array(20).fill(1), ...Array(31).fill(0)])
  const pcm = Float32Array.from({ length: 961 * 512 }, (_, i) => i / (961 * 512))
  const [first, second] = ends(await vad.push(pcm))
  expect(first).toMatchObject({ forced: true, startSample: 4 * 512, endSample: 907 * 512, detectedAtSample: 910 * 512 })
  expect(second).toMatchObject({ forced: false, startSample: 907 * 512, endSample: 936 * 512 })
  expect(second.samples).toEqual(pcm.slice(907 * 512, 936 * 512))
})

// 확인: 긴 쉼(25조각) 중에 강제로 끊기면 보통의 끝처럼 다음 마디는 말 앞 6조각부터 시작하고, 그 앞의 조용한
// 13조각은 어느 마디에도 들어가지 않는다. 말소리 조각은 빠지거나 겹치지 않는다(511b7f8에서 이 동작을 고정).
it('drops the silence between clips as a normal end does when a forced split lands in a long pause', async () => {
  // 25 quiet frames when the limit hits: the forced clip keeps six and leaves out 19. Only the last six
  // become the next pre-roll, as after a normal end; the 13 quiet frames before them are in neither clip.
  // No speech frame is dropped or sent twice.
  const vad = await detector([...Array(10).fill(0), ...Array(875).fill(1), ...Array(25).fill(0),
    ...Array(20).fill(1), ...Array(31).fill(0)])
  const pcm = Float32Array.from({ length: 961 * 512 }, (_, i) => i / (961 * 512))
  const [first, second] = ends(await vad.push(pcm))
  expect(first).toMatchObject({ forced: true, startSample: 4 * 512, endSample: 891 * 512, detectedAtSample: 910 * 512 })
  expect(second).toMatchObject({ forced: false, startSample: 904 * 512, endSample: 936 * 512 })
  expect(first.samples).toEqual(pcm.slice(4 * 512, 891 * 512))
  expect(second.samples).toEqual(pcm.slice(904 * 512, 936 * 512))
})

// 확인: push를 기다리지 않고 연달아 불러도 모델 호출은 순서대로 돌고, 두 번째 호출은 첫 호출이 낸 상태(h =
// 1)를 받는다.
// 왜: 마이크 조각은 앞 처리가 끝나기 전에 계속 들어온다. 동시에 돌면 순환 상태가 꼬여 확률이 틀어진다.
it('serializes concurrent chunks rather than running frames with stale state', async () => {
  const vad = await detector([1, 1])
  const events = await Promise.all([vad.push(samples(1)), vad.push(samples(1))])
  expect(events[1][0].type).toBe('start')
  expect(ort.run.mock.calls[1][0].h.data[0]).toBe(1)
})

// 확인: 모델 실행 중에 interrupt(재생 시작 등)하면, 실행 중이던 결과와 대기 중이던 조각은 이벤트 없이
// 버려지고, 다음 조각은 0으로 초기화된 앞 64표본과 상태로 시작한다.
// 왜: 재생 전의 상태가 재생 뒤의 첫 판정에 섞이면 엉뚱한 시작·끝이 나올 수 있다.
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

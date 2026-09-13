import { env, InferenceSession, Tensor } from 'onnxruntime-web/wasm'
import wasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url'
import wasmModuleUrl from 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url'
import modelUrl from './assets/silero_vad_v6.onnx?url'
import type { VadRunner } from './silero'

let loading: Promise<VadRunner> | undefined

export function getSileroRunner(): Promise<VadRunner> {
  loading ??= load().catch((error: unknown) => { loading = undefined; throw error })
  return loading
}

async function load(): Promise<VadRunner> {
  env.wasm.numThreads = 1 // Also works without cross-origin isolation.
  env.wasm.wasmPaths = { wasm: wasmUrl, mjs: wasmModuleUrl }
  const session = await InferenceSession.create(modelUrl, { executionProviders: ['wasm'] })
  // A single lazily loaded session survives stop/start. Serialize runs across old/new detectors too.
  let tail: Promise<unknown> = Promise.resolve()
  return (input, h, c) => {
    const task = tail.then(async () => {
      const out = await session.run({
        input: new Tensor('float32', input, [1, 576]),
        h: new Tensor('float32', h, [1, 1, 128]),
        c: new Tensor('float32', c, [1, 1, 128]),
      })
      return { probability: Number(out.speech_probs.data[0]),
        h: Float32Array.from(out.hn.data as Float32Array), c: Float32Array.from(out.cn.data as Float32Array) }
    })
    tail = task.catch(() => undefined)
    return task
  }
}

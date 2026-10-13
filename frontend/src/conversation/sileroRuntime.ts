// [파일 역할] Silero VAD 모델을 브라우저에서 실제로 돌리는 부분. onnxruntime-web(WASM)으로 ONNX 모델 세션을
// 한 번 만들고, silero.ts가 쓰는 VadRunner 함수를 돌려준다.
// [설계 이유] silero.ts가 이 파일을 동적 import로만 불러, 런타임과 모델은 대화 기능을 처음 시작할 때 한 번만
// 받는다. 모델은 faster-whisper에 들어 있는 silero_vad_v6.onnx를 그대로 복사한 것이라 서버 인식의 VAD와 같은
// 파일이다(src/conversation/assets/SOURCE.md). 브라우저 접근성 검사는 이 모듈 주소를 가짜로 바꿔 모델 없이
// 화면을 검사한다(e2e/accessibility.spec.ts).
// [import] onnxruntime-web/wasm은 WASM 실행만 담은 진입점이다. ?url이 붙은 세 줄은 Vite가 WASM, 그 로더(mjs),
// 모델 파일을 해시 붙은 로컬 자산으로 내보내고 주소만 주게 한다. 실행 중에 CDN이나 외부 서버에서 받지
// 않는다(vite.config.ts의 해석 조건과 짝).

import { env, InferenceSession, Tensor } from 'onnxruntime-web/wasm'
import wasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url'
import wasmModuleUrl from 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url'
import modelUrl from './assets/silero_vad_v6.onnx?url'
import type { VadRunner } from './silero'

// 만들고 있거나 만든 세션의 약속. 페이지에 하나만 둔다.
let loading: Promise<VadRunner> | undefined

// 세션을 처음 부를 때만 만들고 이후에는 같은 약속을 돌려준다. 실패하면 기억을 지워, 사용자가 대화를 다시
// 시작할 때 처음부터 다시 받게 한다.
export function getSileroRunner(): Promise<VadRunner> {
  loading ??= load().catch((error: unknown) => { loading = undefined; throw error })
  return loading
}

// 세션 만들기와 실행 함수 정의.
async function load(): Promise<VadRunner> {
  env.wasm.numThreads = 1 // Also works without cross-origin isolation.
  // 런타임이 WASM과 로더를 찾을 주소를 위의 로컬 자산 주소로 정한다.
  env.wasm.wasmPaths = { wasm: wasmUrl, mjs: wasmModuleUrl }
  // WASM(CPU) 실행 제공자로 모델 세션을 만든다. 32ms 조각 하나를 처리하는 작은 모델이다.
  const session = await InferenceSession.create(modelUrl, { executionProviders: ['wasm'] })
  // A single lazily loaded session survives stop/start. Serialize runs across old/new detectors too.
  let tail: Promise<unknown> = Promise.resolve()
  return (input, h, c) => {
    // 입력 텐서: input [1, 576](앞 64 + 이번 512표본), h·c [1, 1, 128](순환 상태). 출력: speech_probs(말소리
    // 확률), hn·cn(다음 상태). 출력 배열은 런타임이 쓰는 버퍼일 수 있어 Float32Array.from으로 복사해 넘긴다.
    const task = tail.then(async () => {
      const out = await session.run({
        input: new Tensor('float32', input, [1, 576]),
        h: new Tensor('float32', h, [1, 1, 128]),
        c: new Tensor('float32', c, [1, 1, 128]),
      })
      return { probability: Number(out.speech_probs.data[0]),
        h: Float32Array.from(out.hn.data as Float32Array), c: Float32Array.from(out.cn.data as Float32Array) }
    })
    // 한 번의 실패가 다음 실행 순서를 막지 않게 tail에는 오류를 삼킨 약속을 둔다. 실패 자체는 돌려준 task로
    // 호출 쪽에 전해진다.
    tail = task.catch(() => undefined)
    return task
  }
}

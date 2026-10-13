// [파일 역할] 마이크를 열어 16kHz 모노 float 소리 조각을 콜백으로 계속 넘겨 주는 함수. 대화 모드·동시통역·두
// 사람 대화 세션이 같은 함수를 쓴다.
// [흐름] AudioContext 만들기 → 사용자 제스처 안에서 resume → getUserMedia로 마이크 권한·스트림 받기 →
// AudioWorklet(capture.worklet.js) 올리기 → 마이크 → 워클릿 연결 → 워클릿이 보낸 조각을 16kHz로 바꿔 samples
// 콜백에 넘김.
// [설계 이유] 열기 도중 어느 단계에서든 취소(signal)되거나 실패하면 stop 하나로 스트림·노드·컨텍스트를 모두
// 정리해 마이크가 켜진 채 남지 않게 한다.
// [연결] 세션 생성자의 기본값으로 쓰이고(conversation·live·dialog의 session.ts), 테스트는 OpenMicrophone
// 타입에 맞는 가짜 함수를 넣는다. ?url 접미는 Vite가 워클릿 파일을 별도 자산으로 내보내고 그 주소를 주게
// 한다(addModule은 주소가 필요하다).

import workletUrl from './capture.worklet.js?url'
import { MonoResampler } from './pcm'

// 열린 마이크: stop은 완전히 닫기, setPaused는 번역 음성을 재생하는 동안 잠시 듣지 않기(재생 소리를 다시 말로
// 듣지 않도록).
export interface Microphone { stop(): void; setPaused(paused: boolean): void }
// samples: 16kHz 조각을 받을 함수, signal: 열기·사용 중 취소, failed: 마이크가 도중에 끊기거나 워클릿이
// 오류를 냈을 때 불릴 함수.
export type OpenMicrophone = (samples: (chunk: Float32Array) => void, signal: AbortSignal, failed: () => void) => Promise<Microphone>

export const openMicrophone: OpenMicrophone = async (samples, signal, failed) => {
  // 보안 컨텍스트(https 또는 localhost)가 아니거나 오래된 브라우저면 이 기능이 없다. 세션이 이 오류를 받아
  // 안내 문구를 띄운다.
  if (!navigator.mediaDevices?.getUserMedia || !window.AudioContext) throw new Error('Microphone unavailable')
  const context = new AudioContext()
  let stream: MediaStream | undefined
  let source: MediaStreamAudioSourceNode | undefined
  let node: AudioWorkletNode | undefined
  let closed = false
  let paused = false
  let epoch = 0
  // 정리 함수. 여러 번 불려도 한 번만 정리한다. 트랙을 멈춰 마이크를 놓고(onended를 먼저 떼어 정리 중에
  // failed가 불리지 않게), 노드 연결과 port를 닫고, AudioContext를 닫는다.
  const stop = () => {
    if (closed) return
    closed = true
    signal.removeEventListener('abort', stop)
    stream?.getTracks().forEach((track) => { track.onended = null; track.stop() })
    source?.disconnect()
    if (node) { node.port.onmessage = null; node.port.close(); node.disconnect() }
    void context.close().catch(() => undefined)
  }
  signal.addEventListener('abort', stop, { once: true })
  try {
    if (signal.aborted) throw new Error('Canceled')
    // Resume during the start button gesture; do not wait for the permission prompt first.
    const resumed = context.resume()
    // Permission may outlive a failed resume; mark the rejection handled immediately.
    void resumed.catch(() => undefined)
    // 모노, 에코 제거, 잡음 억제, 자동 음량 조절을 켠 마이크를 요청한다. 말소리 판정(Silero)과 서버 인식이
    // 받는 소리를 브라우저가 한 번 다듬어 준다. 첫 사용이면 여기서 권한 창이 뜬다.
    const acquired = navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } })
    stream = await acquired
    // 권한 창을 기다리는 사이 멈췄으면 방금 받은 스트림도 바로 놓는다.
    if (closed) { stream.getTracks().forEach((track) => track.stop()); throw new Error('Canceled') }
    await resumed
    // 마이크가 뽑히거나 권한이 거둬지면 트랙이 끝난다. 그때 정리하고 세션에 실패를 알린다.
    stream.getTracks().forEach((track) => { track.onended = () => { stop(); failed() } })
    await context.audioWorklet.addModule(workletUrl)
    if (closed) throw new Error('Canceled')
    // AudioContext의 실제 표본화율(보통 44.1·48kHz)에서 16kHz로 바꾸는 변환기. Silero VAD와 서버 인식은
    // 16kHz를 쓴다.
    const resampler = new MonoResampler(context.sampleRate)
    node = new AudioWorkletNode(context, 'conversation-capture')
    node.onprocessorerror = () => { stop(); failed() }
    // 워클릿 조각 받기. 닫혔거나, 일시정지 중이거나, 지금 epoch가 아닌(일시정지 전환 전에 출발한) 조각은
    // 버린다.
    node.port.onmessage = ({ data }: MessageEvent<{ samples: Float32Array; epoch: number }>) => {
      if (!closed && !paused && data.epoch === epoch) samples(resampler.push(data.samples))
    }
    // 마이크 → 워클릿 → 출력(destination)으로 잇는다. 워클릿이 출력을 무음으로 두므로 마이크 소리가 스피커로
    // 나가지 않는다.
    source = context.createMediaStreamSource(stream)
    source.connect(node)
    node.connect(context.destination)
    // setPaused: epoch를 올리고 변환기에 남은 소수 표본을 비운 뒤 워클릿에 알린다. 재생 전후의 소리가 한 조각
    // 안에서 이어 붙지 않게 하려는 것이다.
    return { stop, setPaused(value) {
      paused = value
      epoch++
      resampler.reset()
      node?.port.postMessage({ paused, epoch })
    } }
  // 어느 단계에서 실패하든 연 자원을 모두 닫고 오류를 세션으로 넘긴다.
  } catch (error) { stop(); throw error }
}

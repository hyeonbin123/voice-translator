import workletUrl from './capture.worklet.js?url'
import { MonoResampler } from './pcm'

export interface Microphone { stop(): void; setPaused(paused: boolean): void }
export type OpenMicrophone = (samples: (chunk: Float32Array) => void, signal: AbortSignal, failed: () => void) => Promise<Microphone>

export const openMicrophone: OpenMicrophone = async (samples, signal, failed) => {
  if (!navigator.mediaDevices?.getUserMedia || !window.AudioContext) throw new Error('Microphone unavailable')
  const context = new AudioContext()
  let stream: MediaStream | undefined
  let source: MediaStreamAudioSourceNode | undefined
  let node: AudioWorkletNode | undefined
  let closed = false
  let paused = false
  let epoch = 0
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
    const acquired = navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } })
    stream = await acquired
    if (closed) { stream.getTracks().forEach((track) => track.stop()); throw new Error('Canceled') }
    await resumed
    stream.getTracks().forEach((track) => { track.onended = () => { stop(); failed() } })
    await context.audioWorklet.addModule(workletUrl)
    if (closed) throw new Error('Canceled')
    const resampler = new MonoResampler(context.sampleRate)
    node = new AudioWorkletNode(context, 'conversation-capture')
    node.onprocessorerror = () => { stop(); failed() }
    node.port.onmessage = ({ data }: MessageEvent<{ samples: Float32Array; epoch: number }>) => {
      if (!closed && !paused && data.epoch === epoch) samples(resampler.push(data.samples))
    }
    source = context.createMediaStreamSource(stream)
    source.connect(node)
    node.connect(context.destination)
    return { stop, setPaused(value) {
      paused = value
      epoch++
      resampler.reset()
      node?.port.postMessage({ paused, epoch })
    } }
  } catch (error) { stop(); throw error }
}

import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { useRecorder, MAX_RECORDING_MS } from './useRecorder'
import { MAX_AUDIO_BYTES } from './api'

class Recorder {
  static instances: Recorder[] = []
  static isTypeSupported = (type: string) => type === 'audio/webm;codecs=opus'
  state = 'inactive'
  mimeType = 'audio/webm;codecs=opus'
  ondataavailable: ((event: { data: Blob }) => void) | null = null
  onstop: (() => void) | null = null
  onerror: (() => void) | null = null
  stop = vi.fn(() => {
    this.state = 'inactive'
    // Real MediaRecorder queues final data and stop events, even after a cancel.
    const data = this.ondataavailable; const stop = this.onstop
    queueMicrotask(() => { data?.({ data: new Blob(['recorded audio']) }); stop?.() })
  })
  start = vi.fn(() => { this.state = 'recording' })
  constructor() { Recorder.instances.push(this) }
}
const trackStop = vi.fn()
const stream = { getTracks: () => [{ stop: trackStop }] } as unknown as MediaStream
const getUserMedia = vi.fn<() => Promise<MediaStream>>()
const recorded = vi.fn<(audio: Blob) => void>()
const originalDevices = Object.getOwnPropertyDescriptor(navigator, 'mediaDevices')

beforeEach(() => {
  vi.useFakeTimers()
  Recorder.instances = []
  trackStop.mockReset(); recorded.mockReset(); getUserMedia.mockReset()
  getUserMedia.mockResolvedValue(stream)
  Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getUserMedia } })
  vi.stubGlobal('MediaRecorder', Recorder)
})
afterEach(() => {
  cleanup()
  vi.useRealTimers(); vi.unstubAllGlobals()
  if (originalDevices) Object.defineProperty(navigator, 'mediaDevices', originalDevices)
  else Reflect.deleteProperty(navigator, 'mediaDevices')
})

it('records until stopped, releases the microphone, and returns a typed blob', async () => {
  const { result } = renderHook(() => useRecorder(recorded))
  await act(async () => { await result.current.start() })
  expect(getUserMedia).toHaveBeenCalledWith({ audio: true })
  expect(result.current.state).toBe('recording')
  await act(async () => { result.current.stop() })
  expect(result.current.state).toBe('idle')
  expect(recorded).toHaveBeenCalledTimes(1)
  expect(recorded.mock.calls[0][0].type).toBe('audio/webm;codecs=opus')
  expect(recorded.mock.calls[0][0].size).toBeGreaterThan(0)
  expect(trackStop).toHaveBeenCalledOnce()
  expect(vi.getTimerCount()).toBe(0)
})

it('automatically stops at 30 seconds and announces the limit', async () => {
  const { result } = renderHook(() => useRecorder(recorded))
  await act(async () => { await result.current.start() })
  await act(async () => { await vi.advanceTimersByTimeAsync(MAX_RECORDING_MS - 1) })
  expect(result.current.state).toBe('recording')
  expect(recorded).not.toHaveBeenCalled()
  await act(async () => { await vi.advanceTimersByTimeAsync(1) })
  expect(result.current.state).toBe('idle')
  expect(result.current.notice).toContain('30초 제한')
  expect(recorded).toHaveBeenCalledOnce()
  expect(trackStop).toHaveBeenCalledOnce()
})

it.each(['cancel', 'unmount'] as const)('discards recording events and releases tracks on %s', async (action) => {
  const { result, unmount } = renderHook(() => useRecorder(recorded))
  await act(async () => { await result.current.start() })
  await act(async () => {
    if (action === 'cancel') result.current.cancel()
    else unmount()
  })
  expect(recorded).not.toHaveBeenCalled()
  expect(trackStop).toHaveBeenCalledOnce()
  expect(Recorder.instances[0].stop).toHaveBeenCalledOnce()
  expect(vi.getTimerCount()).toBe(0)
})

it.each(['cancel', 'unmount'] as const)('stops a late permission stream after %s without creating a recorder', async (action) => {
  let resolve!: (value: MediaStream) => void
  getUserMedia.mockReturnValueOnce(new Promise((done) => { resolve = done }))
  const { result, unmount } = renderHook(() => useRecorder(recorded))
  act(() => { void result.current.start() })
  expect(result.current.state).toBe('permission')
  act(() => { if (action === 'cancel') result.current.cancel(); else unmount() })
  await act(async () => { resolve(stream) })
  expect(Recorder.instances).toHaveLength(0)
  expect(trackStop).toHaveBeenCalledOnce()
  expect(recorded).not.toHaveBeenCalled()
})

it('handles permission denial and permits a fresh attempt', async () => {
  getUserMedia.mockRejectedValueOnce(new DOMException('denied', 'NotAllowedError'))
  const { result } = renderHook(() => useRecorder(recorded))
  await act(async () => { await result.current.start() })
  expect(result.current.state).toBe('idle')
  expect(result.current.error).toContain('권한이 거부')
  await act(async () => { await result.current.start() })
  expect(result.current.state).toBe('recording')
  expect(result.current.error).toBe('')
})

it('discards recordings that exceed the byte limit', async () => {
  const { result } = renderHook(() => useRecorder(recorded))
  await act(async () => { await result.current.start() })
  const chunk = new Blob(['x'])
  Object.defineProperty(chunk, 'size', { value: MAX_AUDIO_BYTES + 1 })
  await act(async () => { Recorder.instances[0].ondataavailable?.({ data: chunk }) })
  expect(result.current.state).toBe('idle')
  expect(result.current.error).toContain('10MB')
  expect(recorded).not.toHaveBeenCalled()
  expect(trackStop).toHaveBeenCalledOnce()
})

it('releases resources after recorder errors', async () => {
  const { result } = renderHook(() => useRecorder(recorded))
  await act(async () => { await result.current.start() })
  await act(async () => { Recorder.instances[0].onerror?.() })
  expect(result.current.error).toContain('녹음 중 오류')
  expect(result.current.state).toBe('idle')
  expect(recorded).not.toHaveBeenCalled()
  expect(trackStop).toHaveBeenCalledOnce()
})

it('offers file input when MediaRecorder is unavailable', async () => {
  vi.stubGlobal('MediaRecorder', undefined)
  const { result } = renderHook(() => useRecorder(recorded))
  await act(async () => { await result.current.start() })
  expect(result.current.error).toContain('음성 파일을 선택')
  expect(getUserMedia).not.toHaveBeenCalled()
})

// [파일 역할] 음성 입력 녹음 훅(useRecorder) 테스트 묶음. 녹음 시작·끝·취소·언마운트, 29초 자동 정지, 권한
// 거부, 크기 초과, 녹음기 오류, 지원하지 않는 브라우저에서 상태와 자원 정리를 확인한다.
// [가짜 객체] MediaRecorder를 가짜 클래스로 바꾸고 getUserMedia를 조종한다. 가짜 녹음기의 stop은 실제처럼
// 마지막 데이터와 stop 이벤트를 다음 마이크로태스크에 보낸다. 그래서 취소 뒤에도 늦게 오는 이벤트를 훅이
// 무시하는지 확인할 수 있다. 타이머는 가짜 타이머로 시간을 앞당긴다.

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

// 확인: 끝내기를 누르면 webm/Opus 형식의 비어 있지 않은 Blob이 한 번 넘어가고, 마이크 트랙을 놓으며 남은
// 타이머가 없다.
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

// 확인: 자동 정지 시각에 1초 늦음을 더해도 서버 한도 30초를 넘지 않는다. 숨겨진 탭의 늦은 타이머 때문에 422가
// 나던 문제(dfc251d)를 숫자로 고정한다.
it('stops early enough that a timer up to a second late stays within the server 30 s limit', () => {
  // Hidden pages wake timers about once per second, so the stop can come up to 1 s late.
  expect(MAX_RECORDING_MS + 1000).toBeLessThanOrEqual(30_000)
})

// 확인: 시작 29초 뒤에는 녹음이 멈추고 결과가 넘어가 있다.
it('has stopped and delivered the recording 29 seconds after start', async () => {
  const { result } = renderHook(() => useRecorder(recorded))
  await act(async () => { await result.current.start() })
  await act(async () => { await vi.advanceTimersByTimeAsync(29_000) })
  expect(result.current.state).toBe('idle')
  expect(recorded).toHaveBeenCalledOnce()
})

// 확인: 한도 1ms 전까지는 녹음 중이고, 한도에서 멈추며 '30초 제한' 안내와 함께 결과를 넘기고 마이크를 놓는다.
it('automatically stops before the server 30 s limit and announces the limit', async () => {
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

// 확인: 녹음 중 취소하거나 화면을 떠나면, 녹음기가 늦게 보내는 데이터·stop 이벤트가 와도 결과를 넘기지 않고
// 마이크와 타이머를 정리한다.
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

// 확인: 권한 창을 기다리는 중에 취소하거나 떠난 뒤 권한이 허락되면, 녹음기를 만들지 않고 받은 마이크를 바로
// 놓는다.
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

// 확인: 권한 거부(NotAllowedError)면 거부 안내를 보이고 대기로 돌아가며, 다시 시작하면 오류를 지우고
// 녹음한다.
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

// 확인: 녹음 데이터가 크기 한도를 넘으면 녹음을 버리고 10MB 안내를 보이며 마이크를 놓는다.
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

// 확인: 녹음기 오류가 나면 오류를 보이고 결과 없이 대기로 돌아가며 마이크를 놓는다.
it('releases resources after recorder errors', async () => {
  const { result } = renderHook(() => useRecorder(recorded))
  await act(async () => { await result.current.start() })
  await act(async () => { Recorder.instances[0].onerror?.() })
  expect(result.current.error).toContain('녹음 중 오류')
  expect(result.current.state).toBe('idle')
  expect(recorded).not.toHaveBeenCalled()
  expect(trackStop).toHaveBeenCalledOnce()
})

// 확인: MediaRecorder가 없는 브라우저에서는 마이크를 요청하지 않고 음성 파일 선택을 안내한다.
it('offers file input when MediaRecorder is unavailable', async () => {
  vi.stubGlobal('MediaRecorder', undefined)
  const { result } = renderHook(() => useRecorder(recorded))
  await act(async () => { await result.current.start() })
  expect(result.current.error).toContain('음성 파일을 선택')
  expect(getUserMedia).not.toHaveBeenCalled()
})

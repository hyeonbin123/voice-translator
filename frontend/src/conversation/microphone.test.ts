import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { openMicrophone } from './microphone'

const stopTrack = vi.fn(), disconnect = vi.fn(), resume = vi.fn(), close = vi.fn(), addModule = vi.fn()
const track = { stop: stopTrack, onended: null as (() => void) | null }
const getUserMedia = vi.fn()
let port: { onmessage: ((event: { data: { samples: Float32Array; epoch: number } }) => void) | null; close: ReturnType<typeof vi.fn>; postMessage: ReturnType<typeof vi.fn> }
let processorError: (() => void) | undefined
beforeEach(() => {
  vi.clearAllMocks(); resume.mockResolvedValue(undefined); close.mockResolvedValue(undefined); addModule.mockResolvedValue(undefined)
  getUserMedia.mockResolvedValue({ getTracks: () => [track] })
  Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getUserMedia } })
  vi.stubGlobal('AudioContext', class {
    sampleRate = 48_000; resume = resume; close = close; audioWorklet = { addModule }; destination = {}
    createMediaStreamSource() { return { connect: vi.fn(), disconnect } }
  })
  vi.stubGlobal('AudioWorkletNode', class {
    port = port = { onmessage: null, close: vi.fn(), postMessage: vi.fn() }
    connect = vi.fn(); disconnect = disconnect
    set onprocessorerror(value: () => void) { processorError = value }
  })
})
afterEach(() => vi.unstubAllGlobals())

it('opens echo-canceled mono capture with automatic gain, resamples, drops paused/stale worklet messages and releases resources', async () => {
  const capture = vi.fn(), controller = new AbortController()
  const mic = await openMicrophone(capture, controller.signal, vi.fn())
  expect(getUserMedia).toHaveBeenCalledWith({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } })
  port.onmessage!({ data: { samples: new Float32Array(480), epoch: 0 } })
  expect(capture.mock.calls[0][0].length).toBe(160)
  mic.setPaused(true)
  port.onmessage!({ data: { samples: new Float32Array(480), epoch: 0 } })
  mic.setPaused(false)
  port.onmessage!({ data: { samples: new Float32Array(480), epoch: 1 } })
  expect(capture).toHaveBeenCalledOnce()
  port.onmessage!({ data: { samples: new Float32Array(480), epoch: 2 } })
  expect(capture).toHaveBeenCalledTimes(2)
  controller.abort(); mic.stop()
  expect(stopTrack).toHaveBeenCalledOnce(); expect(close).toHaveBeenCalledOnce()
  expect(port.onmessage).toBeNull(); expect(port.close).toHaveBeenCalledOnce()
})

it('releases permission granted after stop without installing a worklet', async () => {
  let grant!: (value: unknown) => void
  getUserMedia.mockImplementationOnce(() => new Promise((resolve) => { grant = resolve }))
  const controller = new AbortController()
  const pending = openMicrophone(vi.fn(), controller.signal, vi.fn())
  controller.abort(); grant({ getTracks: () => [track] })
  await expect(pending).rejects.toThrow('Canceled')
  expect(stopTrack).toHaveBeenCalledOnce(); expect(addModule).not.toHaveBeenCalled()
})

it.each(['track', 'processor'])('reports %s failure and closes the microphone', async (failure) => {
  const failed = vi.fn()
  await openMicrophone(vi.fn(), new AbortController().signal, failed)
  if (failure === 'track') track.onended!(); else processorError!()
  expect(failed).toHaveBeenCalledOnce(); expect(stopTrack).toHaveBeenCalledOnce()
})

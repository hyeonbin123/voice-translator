// [파일 역할] microphone.ts(openMicrophone) 테스트 묶음. 실제 마이크·오디오 장치 없이, 마이크 요청 옵션,
// 16kHz 변환, 일시정지·오래된 조각 버리기, 자원 정리, 도중 실패 처리를 확인한다.
// [가짜 객체] jsdom에는 Web Audio가 없으므로 AudioContext(표본화율 48kHz), AudioWorkletNode,
// navigator.mediaDevices.getUserMedia를 vi.stubGlobal 등으로 가짜로 바꾼다. 가짜 노드의 port를 바깥 변수에
// 잡아 두어, 테스트가 워클릿이 보낸 것처럼 메시지를 직접 넣는다.

import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { openMicrophone } from './microphone'

const stopTrack = vi.fn(), disconnect = vi.fn(), resume = vi.fn(), close = vi.fn(), addModule = vi.fn()
const track = { stop: stopTrack, onended: null as (() => void) | null }
const getUserMedia = vi.fn()
let port: { onmessage: ((event: { data: { samples: Float32Array; epoch: number } }) => void) | null; close: ReturnType<typeof vi.fn>; postMessage: ReturnType<typeof vi.fn> }
let processorError: (() => void) | undefined
// 테스트마다 가짜 함수 기록을 지우고 기본 동작(성공하는 resume·close·addModule, 트랙 하나를 주는 마이크)을
// 다시 정한다.
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

// 확인: (1) 모노·에코 제거·잡음 억제·자동 음량 조절로 마이크를 요청한다. (2) 48kHz 480표본이 16kHz
// 160표본으로 넘어간다. (3) 일시정지 중 조각과, 일시정지를 풀기 전 번호(epoch)의 조각은 버리고 새 번호 조각만
// 받는다. (4) 멈추면 트랙·컨텍스트·port를 한 번씩 닫는다.
// 왜: 재생 중 소리나 재생 직전에 출발한 조각이 판정기로 들어가면 번역 음성을 다시 말로 인식할 수 있다. 정리가
// 빠지면 마이크가 켜진 채 남는다.
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

// 확인: 권한 창을 기다리는 사이 멈춘 뒤 권한이 허락되면, 받은 트랙을 바로 멈추고 워클릿은 올리지 않으며
// 열기는 'Canceled'로 실패한다.
// 왜: 사용자가 이미 멈췄는데 늦게 온 권한 때문에 마이크가 켜진 채 남는 경쟁 상태를 막는다.
it('releases permission granted after stop without installing a worklet', async () => {
  let grant!: (value: unknown) => void
  getUserMedia.mockImplementationOnce(() => new Promise((resolve) => { grant = resolve }))
  const controller = new AbortController()
  const pending = openMicrophone(vi.fn(), controller.signal, vi.fn())
  controller.abort(); grant({ getTracks: () => [track] })
  await expect(pending).rejects.toThrow('Canceled')
  expect(stopTrack).toHaveBeenCalledOnce(); expect(addModule).not.toHaveBeenCalled()
})

// 확인: 사용 중 마이크 트랙이 끝나거나(뽑힘·권한 취소) 워클릿이 오류를 내면 failed가 한 번 불리고 마이크가
// 닫힌다. 세션이 이것으로 '마이크 연결이 끊겼습니다' 안내를 띄운다.
it.each(['track', 'processor'])('reports %s failure and closes the microphone', async (failure) => {
  const failed = vi.fn()
  await openMicrophone(vi.fn(), new AbortController().signal, failed)
  if (failure === 'track') track.onended!(); else processorError!()
  expect(failed).toHaveBeenCalledOnce(); expect(stopTrack).toHaveBeenCalledOnce()
})

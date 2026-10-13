// [파일 역할] pcm.ts 테스트 묶음: 16kHz 변환기(MonoResampler)와 WAV 만들기(encodeWav)를 확인한다. 서버로 가는
// 소리의 형식이 여기서 정해지므로 숫자를 바이트 단위로 확인한다.

import { expect, it } from 'vitest'
import { MonoResampler, encodeWav } from './pcm'

// 확인: 16·44.1·48kHz의 1초 소리가 정확히 16000표본이 되고, 워클릿처럼 128표본씩 나눠 넣어도 한 번에 넣은
// 결과와 똑같으며, 일정한 값(0.4)은 평균해도 그대로다.
// 왜: 마이크 조각 경계에서 소수 표본을 잃거나 두 번 세면 소리 길이와 판정 경계가 어긋난다. 44.1kHz는 비율이
// 정수가 아니라 이 경우를 꼭 본다.
it.each([16_000, 44_100, 48_000])('resamples %i Hz to 16kHz independently of capture chunk boundaries', (rate) => {
  const input = Float32Array.from({ length: rate }, (_, i) => Math.sin(i * .17) * .3)
  const whole = new MonoResampler(rate).push(input)
  const streaming = new MonoResampler(rate)
  const parts: number[] = []
  for (let i = 0; i < input.length; i += 128) parts.push(...streaming.push(input.slice(i, i + 128)))
  expect(whole.length).toBe(16_000)
  expect(Float32Array.from(parts)).toEqual(whole)
  expect(new MonoResampler(rate).push(new Float32Array(rate).fill(.4))[0]).toBeCloseTo(.4)
})

// 확인: reset 뒤에는 앞에 넣던 표본(1, 1)의 부분 합이 남지 않아 다음 출력이 0이다.
// 왜: 재생으로 듣기를 멈췄다 다시 열 때 재생 전 소리가 다음 마디 첫 표본에 섞이지 않아야 한다.
it('clears fractional resampling history at playback discontinuities', () => {
  const converter = new MonoResampler(48_000)
  converter.push(new Float32Array([1, 1])); converter.reset()
  expect(converter.push(new Float32Array([0, 0, 0]))).toEqual(new Float32Array([0]))
})

// 확인: WAV 머리글(RIFF, 크기 46 = 44 + 10 − 8, PCM 형식 1, 채널 1, 16kHz, 16비트, data 10바이트)과, 범위를
// 넘는 −2·2가 −32768·32767로 잘리고 little-endian으로 쓰이는지.
// 왜: 서버는 업로드를 실제로 디코딩하므로 머리글이 틀리면 'Audio could not be decoded'가 된다. FileReader로
// File을 다시 바이트로 읽어 확인한다.
it('encodes a mono 16kHz PCM16 WAV, clips amplitudes and writes little-endian samples', async () => {
  const file = encodeWav(new Float32Array([-2, -1, 0, 1, 2]))
  const buffer = await new Promise<ArrayBuffer>((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as ArrayBuffer)
    reader.onerror = reject
    reader.readAsArrayBuffer(file)
  })
  const view = new DataView(buffer)
  expect(String.fromCharCode(...new Uint8Array(buffer).slice(0, 4))).toBe('RIFF')
  expect(view.getUint32(4, true)).toBe(46)
  expect(view.getUint16(20, true)).toBe(1); expect(view.getUint16(22, true)).toBe(1)
  expect(view.getUint32(24, true)).toBe(16_000); expect(view.getUint16(34, true)).toBe(16)
  expect(view.getUint32(40, true)).toBe(10)
  expect(Array.from({ length: 5 }, (_, i) => view.getInt16(44 + i * 2, true))).toEqual([-32768, -32768, 0, 32767, 32767])
})

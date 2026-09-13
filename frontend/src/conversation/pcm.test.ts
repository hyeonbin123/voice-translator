import { expect, it } from 'vitest'
import { MonoResampler, encodeWav } from './pcm'

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

it('clears fractional resampling history at playback discontinuities', () => {
  const converter = new MonoResampler(48_000)
  converter.push(new Float32Array([1, 1])); converter.reset()
  expect(converter.push(new Float32Array([0, 0, 0]))).toEqual(new Float32Array([0]))
})

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

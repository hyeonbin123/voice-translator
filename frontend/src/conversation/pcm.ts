import { SAMPLE_RATE } from './endpointer'

// Streaming, area-weighted resampling. Fractional input intervals survive chunk boundaries.
export class MonoResampler {
  private readonly ratio: number
  private weight = 0
  private sum = 0
  constructor(inputRate: number) {
    if (!Number.isFinite(inputRate) || inputRate < SAMPLE_RATE) throw new Error('Unsupported sample rate')
    this.ratio = inputRate / SAMPLE_RATE
  }
  reset() { this.weight = this.sum = 0 }
  push(input: Float32Array): Float32Array {
    const output: number[] = []
    for (const sample of input) {
      let remaining = 1
      while (remaining > 1e-9) {
        const take = Math.min(remaining, this.ratio - this.weight)
        this.sum += sample * take
        this.weight += take
        remaining -= take
        if (this.weight >= this.ratio - 1e-9) {
          output.push(this.sum / this.ratio)
          this.sum = this.weight = 0
        }
      }
    }
    return Float32Array.from(output)
  }
}

export function encodePcm16(samples: Float32Array): ArrayBuffer {
  const buffer = new ArrayBuffer(samples.length * 2)
  const view = new DataView(buffer)
  samples.forEach((sample, i) => {
    const value = Math.max(-1, Math.min(1, sample))
    view.setInt16(i * 2, Math.round(value * (value < 0 ? 32768 : 32767)), true)
  })
  return buffer
}

export function encodeWav(samples: Float32Array): File {
  const buffer = new ArrayBuffer(44 + samples.length * 2)
  const view = new DataView(buffer)
  const ascii = (offset: number, value: string) => {
    for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i))
  }
  ascii(0, 'RIFF'); view.setUint32(4, buffer.byteLength - 8, true); ascii(8, 'WAVE')
  ascii(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true)
  view.setUint16(22, 1, true); view.setUint32(24, SAMPLE_RATE, true)
  view.setUint32(28, SAMPLE_RATE * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true)
  ascii(36, 'data'); view.setUint32(40, samples.length * 2, true)
  new Uint8Array(buffer, 44).set(new Uint8Array(encodePcm16(samples)))
  return new File([buffer], 'conversation.wav', { type: 'audio/wav' })
}

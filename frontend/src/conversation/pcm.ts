// [파일 역할] 소리 형식 변환 도우미 세 가지: 마이크의 높은 표본화율을 16kHz로 낮추는 MonoResampler,
// 웹소켓으로 보낼 16비트 PCM 바이트(encodePcm16), HTTP로 올릴 WAV 파일(encodeWav).
// [연결] MonoResampler는 microphone.ts가, encodePcm16은 live/session.ts와
// conversation/speculative.ts가(웹소켓 바이너리: 16kHz 모노 16비트 little-endian, docs/api.md), encodeWav는
// conversation/session.ts와 dialog/session.ts가(POST /api/translate/speech·dialog 업로드) 쓴다.
// SAMPLE_RATE(16000)는 endpointer.ts에 정의된 것을 함께 쓴다.

import { SAMPLE_RATE } from './endpointer'

// 흘려보내며 하는 면적 가중 리샘플링. 입력 구간의 소수 부분은 조각 경계를 넘어 이어진다.
// 입력 표본 ratio(= 입력 표본화율 ÷ 16000)개마다 출력 표본 하나를 만드는데, 각 입력 표본이 출력 구간에 걸친
// 길이만큼 가중 평균한다(구간 평균이라 간단한 저역 통과 효과도 난다). 48kHz면 3개씩, 44.1kHz면 2.75625개씩
// 묶이고, 소수 부분(weight·sum)은 다음 조각으로 넘겨 조각 경계에서 표본이 어긋나지 않는다. 16kHz보다 낮은
// 입력은 올려 표본화하지 않고 오류로 거부한다.
export class MonoResampler {
  private readonly ratio: number
  private weight = 0
  private sum = 0
  constructor(inputRate: number) {
    if (!Number.isFinite(inputRate) || inputRate < SAMPLE_RATE) throw new Error('Unsupported sample rate')
    this.ratio = inputRate / SAMPLE_RATE
  }
  // 남은 부분 합을 버린다. 마이크 일시정지 전환 때 부른다(microphone.ts).
  reset() { this.weight = this.sum = 0 }
  push(input: Float32Array): Float32Array {
    const output: number[] = []
    for (const sample of input) {
      // 입력 표본 하나(길이 1)를 지금 출력 구간의 남은 자리(ratio − weight)만큼 채우고, 구간이 차면 평균을
      // 내보내고 새 구간을 시작한다. 1e-9는 부동소수 오차로 구간이 끝나지 않는 것을 막는 여유다.
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

// −1~1 float를 16비트 정수로 바꾼다. 범위를 넘는 값은 잘라 내고, 음수는 32768, 양수는 32767을 곱해 정수
// 범위(−32768~32767)를 꽉 채운다. setInt16의 마지막 인자 true는 little-endian으로 쓰라는 뜻이다.
export function encodePcm16(samples: Float32Array): ArrayBuffer {
  const buffer = new ArrayBuffer(samples.length * 2)
  const view = new DataView(buffer)
  samples.forEach((sample, i) => {
    const value = Math.max(-1, Math.min(1, sample))
    view.setInt16(i * 2, Math.round(value * (value < 0 ? 32768 : 32767)), true)
  })
  return buffer
}

// 16kHz·16비트·모노 PCM WAV 파일을 만든다. 44바이트 머리글: RIFF 크기, fmt 덩어리(길이 16, 형식 1 = PCM, 채널
// 1, 표본화율, 초당 바이트 = 16000×2, 블록 정렬 2, 비트 16), data 덩어리 크기 순이다. 그 뒤에 encodePcm16의
// 바이트를 붙인다.
// 대화 모드는 웹소켓으로 보낸 PCM과 이 WAV가 같은 소리가 되게 만들어, 미리 처리한 결과가 업로드한 결과와
// 같도록 한다(docs/experiments.md 11절).
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

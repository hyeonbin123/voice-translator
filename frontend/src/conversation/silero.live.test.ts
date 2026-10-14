// [파일 역할] Silero 판정기의 스트리밍 모드(웹소켓으로 소리를 흘려보내는 모드) 테스트 묶음.
// [핵심 약속] 미리 처리가 결과를 바꾸지 않으려면, pause 때까지 서버로 보낸 소리와 마디가 끝날 때까지 보낸
// 소리가 기본 모드가 만들 WAV의 PCM과 바이트 단위로 같아야 한다(docs/experiments.md 11절, c005648). 이 파일은
// 여러 쉼 모양과 29초 강제 끊김에서 그 약속을 확인한다.

import { expect, it } from 'vitest'
import { SileroEndpointer, type LiveEndpointEvent } from './silero'
import { encodePcm16, encodeWav } from './pcm'

// runner: 정해진 확률 순서를 내는 가짜 모델. read: Blob을 바이트로 읽기. sequence: [조각 수, 확률] 묶음을
// 확률 배열로 펼치기.
const runner = (probabilities: number[]) => {
  let index = 0
  return async () => ({ probability: probabilities[index++] ?? 0, h: new Float32Array(128), c: new Float32Array(128) })
}
const read = (blob: Blob) => new Promise<ArrayBuffer>((resolve) => {
  const reader = new FileReader(); reader.onload = () => resolve(reader.result as ArrayBuffer); reader.readAsArrayBuffer(blob)
})
const sequence = (...parts: [number, number][]) => parts.flatMap(([count, probability]) => Array(count).fill(probability))

// 시나리오: 한 번 쉬고 끝, 쉬었다 다시 말하고 끝, 정확히 6조각 쉼, 말하는 중 강제 끊김, 뒤 여유 뒤 강제 끊김,
// 강제 끊김 뒤 다음 마디(말하는 중·짧은 쉼·긴 쉼).
// 확인 순서: (1) 기본 모드로 같은 입력을 처리해 기대 WAV를 얻는다. (2) 스트리밍 모드에 719표본씩(워클릿
// 조각과 무관한 임의 크기) 나눠 넣는다. (3) start·end·discard 이벤트가 기본 모드와 같은지 본다. (4) 마디마다
// 보낸 PCM을 모아, 마지막 pause 시점의 사본(그 뒤 resume이 없었다면)과 end 시점의 전체가 모두 WAV의
// PCM(머리글 44바이트 뒤)과 같은지 본다.
it.each([
  ['simple pause', sequence([10, 0], [8, .5], [31, 0])],
  ['pause resume pause', sequence([10, 0], [8, .5], [20, 0], [8, 1], [31, 0])],
  ['exact six-frame pause', sequence([6, 0], [8, 1], [6, 0], [8, 1], [31, 0])],
  ['forced end in speech', sequence([10, 0], [910, 1])],
  ['forced end after padding', sequence([10, 0], [882, 1], [25, 0])],
  ['forced end in speech, then the next clip', sequence([10, 0], [920, 1], [31, 0])],
  ['forced end in a pause, then the next clip', sequence([10, 0], [891, 1], [9, 0], [20, 1], [31, 0])],
  ['forced end in a long pause, then the next clip', sequence([10, 0], [875, 1], [25, 0], [20, 1], [31, 0])],
])('streams the exact conversation WAV PCM: %s', async (_name, probabilities) => {
  const input = Float32Array.from({ length: probabilities.length * 512 }, (_, i) => Math.sin(i / 79) * 1.2)
  const normal = new SileroEndpointer(runner(probabilities))
  const live = new SileroEndpointer(runner(probabilities), true)
  const expected = (await normal.push(input)).filter((event) => event.type === 'end')
  const events: LiveEndpointEvent[] = []
  // 워클릿 조각 경계가 어디에 오든 이벤트 순서와 표본이 바뀌면 안 된다(그래서 719표본씩 넣는다).
  for (let i = 0; i < input.length; i += 719) events.push(...await live.push(input.slice(i, i + 719)))
  const legacy = events.filter((event) => ['start', 'end', 'discard'].includes(event.type))
  const reference = new SileroEndpointer(runner(probabilities))
  const metadata = (events: LiveEndpointEvent[]) => events.map((event) => event.type === 'end'
    ? { ...event, samples: event.samples.length } : event)
  expect(metadata(legacy)).toEqual(metadata(await reference.push(input)))
  let sent: number[] = [], snapshot: number[] | null = null, ended = 0
  for (const event of events) {
    if (event.type === 'start') { sent = []; snapshot = null }
    if (event.type === 'audio') sent.push(...new Uint8Array(encodePcm16(event.samples)))
    if (event.type === 'pause') snapshot = [...sent]
    if (event.type === 'resume') snapshot = null
    if (event.type === 'end') {
      const pcm = new Uint8Array(await read(encodeWav(expected[ended++].samples))).slice(44)
      for (const bytes of [snapshot ?? sent, sent]) {
        expect(bytes.length).toBe(pcm.length)
        expect(bytes.every((byte, i) => byte === pcm[i])).toBe(true)
      }
    }
  }
  expect(ended).toBe(expected.length)
  expect(ended).toBeGreaterThan(0)
})

// 확인: 시작하면 앞 여유 6 + 시작 2조각을 audio로 보내고, 말 뒤 6조각째 조용할 때 pause를 쉼마다 한 번씩
// 보내며(이 입력에서는 두 번), 다시 말하면 들고 있던 조용한 조각보다 resume을 먼저 보낸다. 말소리가
// 4조각(128ms)뿐이라 마지막은 discard다.
// 왜: 서버는 resume을 받아야 미리 한 처리를 버리고 이어지는 소리를 같은 마디로 받는다(docs/api.md 메시지
// 순서).
it('emits 6 preroll + 2 onset frames, pause once at 6 quiet frames, resume before buffered audio, and discard', async () => {
  const probabilities = sequence([10, 0], [2, 1], [12, 0], [2, 1], [31, 0])
  const vad = new SileroEndpointer(runner(probabilities), true)
  const events = await vad.push(new Float32Array(probabilities.length * 512))
  expect(events[0]).toEqual({ type: 'start', startSample: 4 * 512 })
  expect(events.slice(1, 15).every((event) => event.type === 'audio')).toBe(true)
  expect(events[15]).toEqual({ type: 'pause' })
  expect(events[16]).toEqual({ type: 'resume' })
  expect(events.at(-1)).toEqual({ type: 'discard' })
  expect(events.filter((event) => event.type === 'pause')).toHaveLength(2)
})

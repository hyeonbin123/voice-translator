import { expect, it } from 'vitest'
import { SileroEndpointer, type LiveEndpointEvent } from './silero'
import { encodePcm16, encodeWav } from './pcm'

const runner = (probabilities: number[]) => {
  let index = 0
  return async () => ({ probability: probabilities[index++] ?? 0, h: new Float32Array(128), c: new Float32Array(128) })
}
const read = (blob: Blob) => new Promise<ArrayBuffer>((resolve) => {
  const reader = new FileReader(); reader.onload = () => resolve(reader.result as ArrayBuffer); reader.readAsArrayBuffer(blob)
})
const sequence = (...parts: [number, number][]) => parts.flatMap(([count, probability]) => Array(count).fill(probability))

it.each([
  ['simple pause', sequence([10, 0], [8, .5], [31, 0])],
  ['pause resume pause', sequence([10, 0], [8, .5], [20, 0], [8, 1], [31, 0])],
  ['exact six-frame pause', sequence([6, 0], [8, 1], [6, 0], [8, 1], [31, 0])],
  ['forced end in speech', sequence([10, 0], [910, 1])],
  ['forced end after padding', sequence([10, 0], [882, 1], [25, 0])],
])('streams the exact conversation WAV PCM: %s', async (_name, probabilities) => {
  const input = Float32Array.from({ length: probabilities.length * 512 }, (_, i) => Math.sin(i / 79) * 1.2)
  const normal = new SileroEndpointer(runner(probabilities))
  const live = new SileroEndpointer(runner(probabilities), true)
  const expected = (await normal.push(input)).filter((event) => event.type === 'end')
  const events: LiveEndpointEvent[] = []
  // Arbitrary worklet chunk boundaries must not change event order or samples.
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

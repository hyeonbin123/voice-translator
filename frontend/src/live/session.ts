import type { Direction, TranslationApi, TranslationResult } from '../translation/api'
import { openMicrophone, type Microphone, type OpenMicrophone } from '../conversation/microphone'
import { createLiveEndpointer, type CreateEndpointer, type SpeechEndpointer, type LiveEndpointEvent } from '../conversation/silero'
import { encodePcm16 } from '../conversation/pcm'
import { PLAYBACK_ERROR, translationPlayer, type PlayTranslation } from '../conversation/player'
import { LiveTransport, type OpenSocket } from './socket'

export interface Caption { text: string; stable: number }
export interface LiveTurn {
  id: number
  direction: Direction
  state: 'speaking' | 'waiting' | 'done' | 'error' | 'canceled'
  source: Caption
  translation: Caption
  result?: TranslationResult
  error?: string
  audioError?: string
}
export interface LiveState {
  active: boolean; permission: boolean; ready: boolean; speaking: boolean; playing: boolean
  playingId: number | null; turns: LiveTurn[]; error: string; notice: string; announcement: string
}
const errors: Record<string, string> = {
  'Audio could not be decoded': '음성을 읽을 수 없습니다. 다시 말해 주세요.',
  'Audio is longer than 30 seconds': '음성은 30초 이하로 말해 주세요.',
  'No speech was recognized': '말소리를 찾지 못했습니다. 마이크를 확인해 주세요.',
  'Translation service is unavailable': '번역 서비스를 사용할 수 없습니다. 잠시 후 다시 말해 주세요.',
  'The translation could not be saved': '번역 결과를 저장하지 못했습니다. 다시 말해 주세요.',
}

export class LiveSession {
  private state: LiveState = { active: false, permission: false, ready: false, speaking: false,
    playing: false, playingId: null, turns: [], error: '', notice: '', announcement: '' }
  private listeners = new Set<() => void>()
  private controller = new AbortController()
  private microphone?: Microphone
  private endpointer?: SpeechEndpointer
  private transport?: LiveTransport
  private captureEpoch = 0
  private current: number | null = null
  private nextId = 1
  private direction: Direction = { source_lang: 'ko', target_lang: 'en' }
  private audioJobs: LiveTurn[] = []
  private playbackPending = false
  private play: PlayTranslation
  private readonly api: TranslationApi
  private readonly open: OpenMicrophone
  private readonly create: CreateEndpointer
  private readonly socket?: OpenSocket
  constructor(api: TranslationApi, open: OpenMicrophone = openMicrophone,
    play: PlayTranslation = translationPlayer(api), create: CreateEndpointer = createLiveEndpointer,
    socket?: OpenSocket) { this.api = api; this.open = open; this.play = play; this.create = create; this.socket = socket }

  getSnapshot = () => this.state
  subscribe = (listener: () => void) => { this.listeners.add(listener); return () => { this.listeners.delete(listener) } }
  private update(patch: Partial<LiveState>) {
    this.state = { ...this.state, ...patch }; this.listeners.forEach((listener) => listener())
  }
  private turn(id: number, patch: Partial<LiveTurn>) {
    this.update({ turns: this.state.turns.map((turn) => turn.id === id ? { ...turn, ...patch } : turn) })
  }
  private fail = (error: string) => { this.stop(); this.update({ error }) }

  async start(direction: Direction) {
    this.stop()
    this.direction = { ...direction }
    const signal = this.controller.signal
    this.update({ active: true, permission: true, error: '', notice: '' })
    const transport = new LiveTransport(this.api, { ...direction }, {
      ready: (ready) => {
        if (signal.aborted) return
        if (!ready) {
          this.interrupt()
          this.cancelPending()
          this.update({ notice: '인증을 새로 확인하고 연결하는 중입니다. 듣는 중 표시 뒤 다시 말해 주세요.' })
        }
        this.update({ ready })
        this.microphone?.setPaused(!ready || this.state.playing || this.state.permission)
      },
      message: (message) => { if (!signal.aborted) this.receive(message) },
      error: (error) => { if (!signal.aborted) this.fail(error) },
    }, this.socket)
    this.transport = transport
    let pendingSamples = 0
    try {
      const [endpointer] = await Promise.all([
        this.create().then((detector) => {
          if (signal.aborted) detector.interrupt()
          return detector
        }),
        this.open((samples) => {
          if (signal.aborted || this.state.permission || !this.state.ready || this.state.playing || !this.endpointer) return
          const epoch = this.captureEpoch
          pendingSamples += samples.length
          if (pendingSamples > 48_000) {
            this.fail('말소리 감지가 입력 속도를 따라가지 못해 멈췄습니다. 다시 시작해 주세요.')
            return
          }
          void this.endpointer.push(samples).then((events) => {
            if (signal.aborted || epoch !== this.captureEpoch) return
            for (const event of events) {
              if (signal.aborted || epoch !== this.captureEpoch) break
              this.event(event)
            }
          }).catch(() => {
            if (!signal.aborted && epoch === this.captureEpoch) this.fail('말소리 감지에 실패했습니다. 다시 시작해 주세요.')
          }).finally(() => { pendingSamples -= samples.length })
        }, signal, () => {
          if (!signal.aborted) this.fail('마이크 연결이 끊겼습니다. 연결을 확인하고 다시 시작해 주세요.')
        }).then((microphone) => {
          if (signal.aborted) microphone.stop()
          else { this.microphone = microphone; microphone.setPaused(true) }
        }),
        transport.start(signal),
      ])
      if (signal.aborted) return
      this.endpointer = endpointer
      this.update({ permission: false })
      this.microphone?.setPaused(!this.state.ready || this.state.playing)
    } catch {
      if (!signal.aborted) this.fail('동시통역을 준비하지 못했습니다. 마이크 권한과 HTTPS 연결을 확인해 주세요.')
    }
  }

  private event(event: LiveEndpointEvent) {
    if (event.type === 'start') {
      const id = this.nextId++
      this.current = id
      this.update({ speaking: true, notice: '', turns: [...this.state.turns,
        { id, direction: this.direction, state: 'speaking', source: { text: '', stable: 0 }, translation: { text: '', stable: 0 } }] })
      this.transport?.send({ type: 'utterance', id })
    } else if (this.current !== null) {
      const id = this.current
      if (event.type === 'audio') this.transport?.send(encodePcm16(event.samples))
      else if (event.type === 'pause' || event.type === 'resume') this.transport?.send({ type: event.type, id })
      else if (event.type === 'end') {
        this.current = null
        this.turn(id, { state: 'waiting' })
        this.update({ speaking: false })
        this.transport?.send({ type: 'end', id })
      } else if (event.type === 'discard') {
        this.cancelCurrent()
        this.update({ notice: '짧은 소리는 건너뛰었습니다. 계속 말해 주세요.' })
      }
    }
  }

  private receive(value: unknown) {
    if (!value || typeof value !== 'object' || !('id' in value) || !('type' in value)) return
    const turn = this.state.turns.find((entry) => entry.id === value.id)
    if (!turn || turn.state === 'done' || turn.state === 'error' || turn.state === 'canceled') return
    if ((value.type === 'source' || value.type === 'translation') && 'text' in value && 'stable' in value
      && typeof value.text === 'string' && typeof value.stable === 'number' && Number.isInteger(value.stable)) {
      const length = Array.from(value.text).length // server counts Unicode code points, not UTF-16 units
      this.turn(turn.id, { [value.type]: { text: value.text, stable: Math.max(0, Math.min(length, value.stable)) } })
    } else if (value.type === 'error') {
      const detail = 'detail' in value && typeof value.detail === 'string' ? value.detail : ''
      this.turn(turn.id, { state: 'error', error: errors[detail] ?? '이 말을 번역하지 못했습니다. 다시 말해 주세요.' })
    } else if (value.type === 'final' && 'result' in value && isResult(value.result)) {
      const result = value.result
      this.turn(turn.id, { state: 'done', result, source: { text: result.source_text, stable: Array.from(result.source_text).length },
        translation: { text: result.translated_text, stable: Array.from(result.translated_text).length } })
      this.update({ announcement: `${turn.id}번째 번역 완료. 원문: ${result.source_text} 번역문: ${result.translated_text}` })
      this.audioJobs.push({ ...turn, state: 'done', result })
      void this.drainAudio()
    }
  }

  private cancelCurrent() {
    if (this.current !== null) {
      const id = this.current
      this.current = null
      this.turn(id, { state: 'canceled' })
      this.transport?.send({ type: 'cancel', id })
    }
    this.update({ speaking: false })
  }
  private interrupt() {
    this.captureEpoch++
    this.endpointer?.interrupt()
    this.cancelCurrent()
  }
  private cancelPending() {
    this.update({ turns: this.state.turns.map((turn) => turn.state === 'waiting' || turn.state === 'speaking'
      ? { ...turn, state: 'canceled' } : turn) })
  }
  stop = () => {
    this.interrupt()
    for (const turn of this.state.turns) if (turn.state === 'waiting') this.transport?.send({ type: 'cancel', id: turn.id })
    this.controller.abort(); this.controller = new AbortController()
    this.transport?.stop(); this.transport = undefined
    this.microphone?.stop(); this.microphone = undefined
    this.endpointer = undefined
    this.audioJobs = []; this.playbackPending = false
    this.cancelPending()
    this.update({ active: false, permission: false, ready: false, speaking: false, playing: false, playingId: null, notice: '' })
  }
  replay(id: number) {
    const turn = this.state.turns.find((entry) => entry.id === id)
    if (!turn?.result || this.state.playingId === id || this.audioJobs.some((entry) => entry.id === id)) return
    this.audioJobs.push(turn)
    void this.drainAudio()
  }
  private async drainAudio() {
    if (this.playbackPending) return
    this.playbackPending = true
    const signal = this.controller.signal
    while (this.audioJobs.length && !signal.aborted) {
      const turn = this.audioJobs.shift()!
      this.turn(turn.id, { audioError: undefined })
      this.update({ playingId: turn.id })
      try {
        await this.play(turn.result!, signal, () => {
          if (signal.aborted) return
          const discarded = this.current !== null
          this.interrupt()
          this.microphone?.setPaused(true)
          this.update({ playing: true, ...(discarded ? { notice: '음성 재생으로 아직 끝나지 않은 말은 취소했습니다. 재생 후 다시 말해 주세요.' } : {}) })
        })
      } catch {
        if (!signal.aborted) this.turn(turn.id, { audioError: PLAYBACK_ERROR })
      } finally {
        if (!signal.aborted) {
          this.microphone?.setPaused(!this.state.ready)
          this.update({ playing: false, playingId: null })
        }
      }
    }
    if (!signal.aborted) this.playbackPending = false
  }
}

function isResult(value: unknown): value is TranslationResult {
  if (!value || typeof value !== 'object') return false
  const result = value as Partial<TranslationResult>
  return typeof result.id === 'string' && result.mode === 'speech' && typeof result.source_text === 'string'
    && typeof result.translated_text === 'string' && ['ko', 'en'].includes(result.source_lang ?? '')
    && ['ko', 'en'].includes(result.target_lang ?? '') && (result.audio_id === null || typeof result.audio_id === 'string')
}

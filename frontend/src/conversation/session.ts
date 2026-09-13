import { errorMessage } from '../api/client'
import type { Direction, TranslationApi, TranslationResult } from '../translation/api'
import { createSileroEndpointer, type CreateEndpointer, type SpeechEndpointer } from './silero'
import { openMicrophone, type Microphone, type OpenMicrophone } from './microphone'
import { encodeWav } from './pcm'
import { PLAYBACK_ERROR, translationPlayer, type PlayTranslation } from './player'

export interface Turn {
  id: number
  state: 'queued' | 'translating' | 'done' | 'error' | 'canceled'
  result?: TranslationResult
  error?: string
  audioError?: string
}
export interface ConversationState {
  active: boolean
  permission: boolean
  speaking: boolean
  translating: boolean
  playing: boolean
  playingId: number | null
  turns: Turn[]
  error: string
  notice: string
}
type Job = { id: number; audio: File; direction: Direction }

export class ConversationSession {
  private state: ConversationState = { active: false, permission: false, speaking: false, translating: false,
    playing: false, playingId: null, turns: [], error: '', notice: '' }
  private listeners = new Set<() => void>()
  private controller = new AbortController()
  private microphone?: Microphone
  private endpointer?: SpeechEndpointer
  private captureEpoch = 0
  private direction: Direction = { source_lang: 'ko', target_lang: 'en' }
  private jobs: Job[] = []
  private audioJobs: Turn[] = []
  private requesting = false
  private playbackPending = false
  private nextId = 1
  private readonly api: Pick<TranslationApi, 'speech'>
  private readonly open: OpenMicrophone
  private readonly play: PlayTranslation
  private readonly create: CreateEndpointer

  constructor(api: TranslationApi, open: OpenMicrophone = openMicrophone, play: PlayTranslation = translationPlayer(api),
    create: CreateEndpointer = createSileroEndpointer) {
    this.api = api; this.open = open; this.play = play; this.create = create
  }
  getSnapshot = () => this.state
  subscribe = (listener: () => void) => { this.listeners.add(listener); return () => { this.listeners.delete(listener) } }
  private update(patch: Partial<ConversationState>) {
    this.state = { ...this.state, ...patch }
    this.listeners.forEach((listener) => listener())
  }
  private turn(id: number, patch: Partial<Turn>) {
    this.update({ turns: this.state.turns.map((turn) => turn.id === id ? { ...turn, ...patch } : turn) })
  }

  async start(direction: Direction) {
    this.stop()
    this.direction = { ...direction }
    const signal = this.controller.signal
    this.update({ active: true, permission: true, error: '', notice: '' })
    let pendingSamples = 0
    let openingMicrophone = true
    try {
      const preparing = this.create()
      const opening = this.open((samples) => {
        if (signal.aborted || !this.state.active || this.state.playing || this.state.permission || !this.endpointer) return
        const epoch = this.captureEpoch
        pendingSamples += samples.length
        if (pendingSamples > 16_000 * 3) {
          this.stop()
          this.update({ error: '말소리 감지가 입력 속도를 따라가지 못해 멈췄습니다. 다른 작업을 줄인 뒤 다시 시작해 주세요.' })
          return
        }
        void this.endpointer.push(samples).then((events) => {
          if (signal.aborted || epoch !== this.captureEpoch) return
          for (const event of events) {
            if (event.type === 'start') this.update({ speaking: true, notice: '' })
            else {
              this.update({ speaking: false })
              if (event.type === 'end') this.enqueue(event.samples)
              else this.update({ notice: '짧은 소리는 건너뛰었습니다. 계속 말해 주세요.' })
            }
          }
        }).catch(() => {
          if (signal.aborted || epoch !== this.captureEpoch) return
          this.stop()
          this.update({ error: '말소리 감지에 실패했습니다. 대화를 다시 시작해 주세요.' })
        }).finally(() => { pendingSamples -= samples.length })
      }, signal, () => {
        if (signal.aborted) return
        this.stop()
        this.update({ error: '마이크 연결이 끊겼습니다. 연결을 확인하고 대화를 다시 시작해 주세요.' })
      }).then((microphone) => {
        if (signal.aborted) microphone.stop()
        else this.microphone = microphone
        openingMicrophone = false
      })
      const [endpointer] = await Promise.all([preparing, opening])
      if (signal.aborted) { endpointer.interrupt(); return }
      this.endpointer = endpointer
      this.update({ permission: false })
    } catch {
      if (signal.aborted) return
      this.stop()
      this.update({ error: openingMicrophone
        ? '대화를 준비하지 못했습니다. 마이크 권한과 HTTPS 연결, 브라우저 지원을 확인해 주세요.'
        : '말소리 감지 파일을 준비하지 못했습니다. 연결을 확인하고 다시 시작해 주세요.' })
    }
  }

  stop = () => {
    this.controller.abort()
    this.controller = new AbortController()
    this.microphone?.stop(); this.microphone = undefined
    this.captureEpoch++
    this.endpointer?.interrupt(); this.endpointer = undefined
    this.jobs = []; this.audioJobs = []
    this.requesting = this.playbackPending = false
    this.update({ active: false, permission: false, speaking: false, translating: false, playing: false, playingId: null,
      notice: '', turns: this.state.turns.map((turn) => turn.state === 'queued' || turn.state === 'translating'
        ? { ...turn, state: 'canceled' } : turn) })
  }

  private enqueue(samples: Float32Array) {
    const id = this.nextId++
    this.update({ turns: [...this.state.turns, { id, state: 'queued' }] })
    this.jobs.push({ id, audio: encodeWav(samples), direction: this.direction })
    void this.drainRequests()
  }

  private async drainRequests() {
    if (this.requesting) return
    this.requesting = true
    const signal = this.controller.signal
    while (this.jobs.length && !signal.aborted) {
      const job = this.jobs.shift()!
      this.turn(job.id, { state: 'translating' })
      this.update({ translating: true })
      try {
        const result = await this.api.speech(job.audio, job.direction, signal)
        if (signal.aborted) return
        this.turn(job.id, { state: 'done', result })
        this.audioJobs.push({ id: job.id, state: 'done', result })
        void this.drainAudio()
      } catch (error) {
        if (signal.aborted) return
        this.turn(job.id, { state: 'error', error: errorMessage(error) })
      }
    }
    if (!signal.aborted) { this.requesting = false; this.update({ translating: false }) }
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
          this.captureEpoch++
          const discarded = this.endpointer?.interrupt()
          this.microphone?.setPaused(true)
          this.update({ playing: true, speaking: false,
            ...(discarded ? { notice: '음성 재생으로 아직 끝나지 않은 말은 보내지 않았습니다. 재생 후 다시 말해 주세요.' } : {}) })
        })
      } catch {
        if (signal.aborted) return
        this.turn(turn.id, { audioError: PLAYBACK_ERROR })
      } finally {
        if (!signal.aborted) {
          this.microphone?.setPaused(false)
          this.update({ playing: false, playingId: null })
        }
      }
    }
    if (!signal.aborted) this.playbackPending = false
  }
}

import { errorMessage } from '../api/client'
import type { DialogTranslationResult, Language, TranslationApi, TranslationResult } from '../translation/api'
import { openMicrophone, type Microphone, type OpenMicrophone } from '../conversation/microphone'
import { encodeWav } from '../conversation/pcm'
import { PLAYBACK_ERROR, translationPlayer, type PlayTranslation } from '../conversation/player'
import { createSileroEndpointer, type CreateEndpointer, type SpeechEndpointer } from '../conversation/silero'

export interface DialogTurn {
  id: number
  state: 'queued' | 'translating' | 'done' | 'error' | 'canceled'
  result?: DialogTranslationResult | TranslationResult
  error?: string
  audioError?: string
  correctionError?: string
}
export interface DialogState {
  active: boolean
  permission: boolean
  speaking: boolean
  translating: boolean
  playing: boolean
  playingId: number | null
  correctingId: number | null
  turns: DialogTurn[]
  error: string
  notice: string
}
type Job = { id: number; audio: File }
type AudioJob = { id: number; result: TranslationResult }
const RECORDING_LIMIT = 5

export function wasLanguageGuessed(result: TranslationResult | DialogTranslationResult): result is DialogTranslationResult {
  return 'language_guessed' in result && result.language_guessed
}

export class DialogSession {
  private state: DialogState = { active: false, permission: false, speaking: false, translating: false,
    playing: false, playingId: null, correctingId: null, turns: [], error: '', notice: '' }
  private listeners = new Set<() => void>()
  private controller = new AbortController()
  private microphone?: Microphone
  private endpointer?: SpeechEndpointer
  private captureEpoch = 0
  private jobs: Job[] = []
  private audioJobs: AudioJob[] = []
  private recordings = new Map<number, File>()
  private requesting = false
  private playbackPending = false
  private previousLang?: Language
  private lastProcessedTurnId?: number
  private nextId = 1
  private readonly api: Pick<TranslationApi, 'dialog' | 'speech' | 'removeHistory'>
  private readonly open: OpenMicrophone
  private readonly play: PlayTranslation
  private readonly create: CreateEndpointer

  constructor(api: TranslationApi, open: OpenMicrophone = openMicrophone, play: PlayTranslation = translationPlayer(api),
    create: CreateEndpointer = createSileroEndpointer) {
    this.api = api; this.open = open; this.play = play; this.create = create
  }
  getSnapshot = () => this.state
  subscribe = (listener: () => void) => { this.listeners.add(listener); return () => { this.listeners.delete(listener) } }
  private update(patch: Partial<DialogState>) {
    this.state = { ...this.state, ...patch }
    this.listeners.forEach((listener) => listener())
  }
  private turn(id: number, patch: Partial<DialogTurn>) {
    this.update({ turns: this.state.turns.map((turn) => turn.id === id ? { ...turn, ...patch } : turn) })
  }

  async start() {
    this.stop()
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
            else if (event.type === 'end' || event.type === 'discard') {
              this.update({ speaking: false })
              if (event.type === 'end') this.enqueue(event.samples)
              else this.update({ notice: '짧은 소리는 건너뛰었습니다. 계속 말해 주세요.' })
            }
          }
        }).catch(() => {
          if (signal.aborted || epoch !== this.captureEpoch) return
          this.stop()
          this.update({ error: '말소리 감지에 실패했습니다. 두 사람 대화를 다시 시작해 주세요.' })
        }).finally(() => { pendingSamples -= samples.length })
      }, signal, () => {
        if (signal.aborted) return
        this.stop()
        this.update({ error: '마이크 연결이 끊겼습니다. 연결을 확인하고 두 사람 대화를 다시 시작해 주세요.' })
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
        ? '두 사람 대화를 준비하지 못했습니다. 마이크 권한과 HTTPS 연결, 브라우저 지원을 확인해 주세요.'
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
    this.recordings.clear()
    this.requesting = this.playbackPending = false
    this.previousLang = undefined; this.lastProcessedTurnId = undefined
    this.update({ active: false, permission: false, speaking: false, translating: false, playing: false,
      playingId: null, correctingId: null, notice: '', turns: this.state.turns.map((turn) =>
        turn.state === 'queued' || turn.state === 'translating' ? { ...turn, state: 'canceled' } : turn) })
  }

  private enqueue(samples: Float32Array) {
    const id = this.nextId++
    const audio = encodeWav(samples)
    this.recordings.set(id, audio)
    while (this.recordings.size > RECORDING_LIMIT) this.recordings.delete(this.recordings.keys().next().value!)
    this.update({ turns: [...this.state.turns, { id, state: 'queued' }] })
    this.jobs.push({ id, audio })
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
        const result = await this.api.dialog(job.audio, this.previousLang, signal)
        if (signal.aborted) return
        this.previousLang = result.source_lang
        this.lastProcessedTurnId = job.id
        this.turn(job.id, { state: 'done', result })
        this.audioJobs.push({ id: job.id, result })
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
    this.audioJobs.push({ id, result: turn.result })
    void this.drainAudio()
  }

  async reverse(id: number) {
    const original = this.state.turns.find((entry) => entry.id === id)
    const audio = this.recordings.get(id)
    if (!original?.result || !wasLanguageGuessed(original.result) || !audio || this.state.correctingId !== null || this.state.playingId !== null) return
    const signal = this.controller.signal
    this.turn(id, { correctionError: undefined })
    this.update({ correctingId: id })
    try {
      const replacement = await this.api.speech(audio, {
        source_lang: original.result.target_lang,
        target_lang: original.result.source_lang,
      }, signal)
      if (signal.aborted) return
      await this.api.removeHistory(original.result.id, signal)
      if (signal.aborted) return
      if (this.lastProcessedTurnId === id) this.previousLang = replacement.source_lang
      this.audioJobs = this.audioJobs.filter((entry) => entry.id !== id)
      this.turn(id, { result: replacement, audioError: undefined, correctionError: undefined })
      this.audioJobs.push({ id, result: replacement })
      void this.drainAudio()
    } catch (error) {
      if (!signal.aborted) this.turn(id, { correctionError: `방향을 바꾸지 못했습니다. ${errorMessage(error)}` })
    } finally {
      if (!signal.aborted) this.update({ correctingId: null })
    }
  }

  hasRecording(id: number) { return this.recordings.has(id) }

  private async drainAudio() {
    if (this.playbackPending) return
    this.playbackPending = true
    const signal = this.controller.signal
    while (this.audioJobs.length && !signal.aborted) {
      const job = this.audioJobs.shift()!
      this.turn(job.id, { audioError: undefined })
      this.update({ playingId: job.id })
      try {
        await this.play(job.result, signal, () => {
          if (signal.aborted) return
          this.captureEpoch++
          const discarded = this.endpointer?.interrupt()
          this.microphone?.setPaused(true)
          this.update({ playing: true, speaking: false,
            ...(discarded ? { notice: '음성 재생으로 아직 끝나지 않은 말은 보내지 않았습니다. 재생 후 다시 말해 주세요.' } : {}) })
        })
      } catch {
        if (!signal.aborted) this.turn(job.id, { audioError: PLAYBACK_ERROR })
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

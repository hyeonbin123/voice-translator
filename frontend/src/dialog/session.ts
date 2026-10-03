import { errorMessage } from '../api/client'
import type { DialogTranslationResult, Language, TranslationApi, TranslationResult } from '../translation/api'
import { openMicrophone, type Microphone, type OpenMicrophone } from '../conversation/microphone'
import { encodeWav } from '../conversation/pcm'
import { PLAYBACK_ERROR, translationPlayer, type PlayTranslation } from '../conversation/player'
import { createLiveEndpointer, createSileroEndpointer, type CreateEndpointer, type LiveEndpointEvent,
  type SpeechEndpointer } from '../conversation/silero'
import { CanceledUtterance, RECONNECTING, SpeculativeChannel, type Commit } from '../conversation/speculative'
import { isTranslationResult, type OpenSocket } from '../live/socket'

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
type Job = { id: number; utterance?: number; run: (signal: AbortSignal) => Promise<DialogTranslationResult> }
type AudioJob = { id: number; result: TranslationResult }
const RECORDING_LIMIT = 5

function isDialogResult(value: unknown): value is DialogTranslationResult {
  return isTranslationResult(value) && typeof (value as DialogTranslationResult).language_confidence === 'number'
    && typeof (value as DialogTranslationResult).language_guessed === 'boolean'
}

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
  // Signed in: the live connection that detects and prepares each turn at a pause (T77). Example mode uploads.
  private channel?: SpeculativeChannel<DialogTranslationResult>
  private captureEpoch = 0
  private jobs: Job[] = []
  private audioJobs: AudioJob[] = []
  private recordings = new Map<number, File>()
  private requesting = false
  private playbackPending = false
  private previousLang?: Language
  private lastProcessedTurnId?: number
  private lastProcessedUtterance?: number // its id on the live connection, which keeps the language too
  private nextId = 1
  private readonly api: Pick<TranslationApi, 'dialog' | 'speech' | 'removeHistory' | 'liveToken' | 'expireLiveAuthentication'>
  private readonly open: OpenMicrophone
  private readonly play: PlayTranslation
  private readonly create: CreateEndpointer
  private readonly speculative: boolean
  private readonly socket?: OpenSocket

  constructor(api: TranslationApi, open: OpenMicrophone = openMicrophone, play: PlayTranslation = translationPlayer(api),
    create?: CreateEndpointer, socket?: OpenSocket) {
    this.api = api; this.open = open; this.play = play; this.socket = socket
    this.speculative = !api.demo
    this.create = create ?? (this.speculative ? createLiveEndpointer : createSileroEndpointer)
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
    // A new connection (a new token) starts from the last turn so an unsure next turn is guessed as before.
    const channel = this.speculative ? new SpeculativeChannel(this.api, () => ({ mode: 'dialog',
      ...(this.previousLang && this.lastProcessedUtterance !== undefined
        ? { previous_lang: this.previousLang, previous_id: this.lastProcessedUtterance } : {}) }), {
      ready: (ready) => { if (!signal.aborted) this.connected(ready) },
      error: (text) => { if (!signal.aborted) { this.stop(); this.update({ error: text }) } },
    }, '두 사람 대화', isDialogResult, this.socket) : undefined
    this.channel = channel
    let pendingSamples = 0
    let openingMicrophone = true
    try {
      const preparing = this.create()
      const opening = this.open((samples) => {
        if (signal.aborted || !this.state.active || this.state.playing || this.state.permission || !this.endpointer
          || (channel && !channel.ready)) return
        const epoch = this.captureEpoch
        pendingSamples += samples.length
        if (pendingSamples > 16_000 * 3) {
          this.stop()
          this.update({ error: '말소리 감지가 입력 속도를 따라가지 못해 멈췄습니다. 다른 작업을 줄인 뒤 다시 시작해 주세요.' })
          return
        }
        void this.endpointer.push(samples).then((events) => {
          for (const event of events) {
            if (signal.aborted || epoch !== this.captureEpoch) return
            this.event(event)
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
        else { this.microphone = microphone; if (channel) microphone.setPaused(true) }
        openingMicrophone = false
      })
      const [endpointer] = await Promise.all([preparing, opening, channel?.start(signal)])
      if (signal.aborted) { endpointer.interrupt(); return }
      this.endpointer = endpointer
      this.update({ permission: false })
      if (channel) this.microphone?.setPaused(!channel.ready || this.state.playing)
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
    this.channel?.stop(); this.channel = undefined
    this.microphone?.stop(); this.microphone = undefined
    this.captureEpoch++
    this.endpointer?.interrupt(); this.endpointer = undefined
    this.jobs = []; this.audioJobs = []
    this.recordings.clear()
    this.requesting = this.playbackPending = false
    this.previousLang = undefined; this.lastProcessedTurnId = undefined; this.lastProcessedUtterance = undefined
    this.update({ active: false, permission: false, speaking: false, translating: false, playing: false,
      playingId: null, correctingId: null, notice: '', turns: this.state.turns.map((turn) =>
        turn.state === 'queued' || turn.state === 'translating' ? { ...turn, state: 'canceled' } : turn) })
  }

  private event(event: LiveEndpointEvent) {
    if (event.type === 'start') { this.channel?.event(event); this.update({ speaking: true, notice: '' }) }
    else if (event.type === 'end') {
      this.update({ speaking: false })
      if (!this.channel) this.enqueue(event.samples)
      else {
        const commit = this.channel.event(event)
        if (commit) this.enqueue(event.samples, commit)
      }
    } else if (event.type === 'discard') {
      this.channel?.event(event)
      this.update({ speaking: false, notice: '짧은 소리는 건너뛰었습니다. 계속 말해 주세요.' })
    } else this.channel?.event(event)
  }

  /** The live connection is ready again, or reconnecting with a new token (it dropped what it held). */
  private connected(ready: boolean) {
    if (!ready) {
      this.captureEpoch++
      this.endpointer?.interrupt()
      this.update({ speaking: false, notice: RECONNECTING })
    }
    this.microphone?.setPaused(!ready || this.state.playing || this.state.permission)
  }

  private enqueue(samples: Float32Array, commit?: Commit<DialogTranslationResult>) {
    const id = this.nextId++
    const audio = encodeWav(samples)
    this.recordings.set(id, audio)
    while (this.recordings.size > RECORDING_LIMIT) this.recordings.delete(this.recordings.keys().next().value!)
    this.update({ turns: [...this.state.turns, { id, state: 'queued' }] })
    // Over HTTP the last turn's language is read when the request goes out, after the turn before is done.
    this.jobs.push(commit ? { id, utterance: commit.utterance, run: () => commit.result }
      : { id, run: (signal) => this.api.dialog(audio, this.previousLang, signal) })
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
        const result = await job.run(signal)
        if (signal.aborted) return
        this.previousLang = result.source_lang
        this.lastProcessedTurnId = job.id
        this.lastProcessedUtterance = job.utterance
        this.turn(job.id, { state: 'done', result })
        this.audioJobs.push({ id: job.id, result })
        void this.drainAudio()
      } catch (error) {
        if (signal.aborted) return
        this.turn(job.id, error instanceof CanceledUtterance ? { state: 'canceled' } : { state: 'error', error: errorMessage(error) })
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
      if (this.lastProcessedTurnId === id) {
        this.previousLang = replacement.source_lang
        // The live connection keeps the last turn's language itself; it takes this only for that turn.
        if (this.lastProcessedUtterance !== undefined) {
          this.channel?.send({ type: 'previous', id: this.lastProcessedUtterance, lang: replacement.source_lang })
        }
      }
      this.audioJobs = this.audioJobs.filter((entry) => entry.id !== id)
      this.turn(id, { result: replacement, audioError: undefined, correctionError: undefined })
      this.audioJobs.push({ id, result: replacement })
      void this.drainAudio()
      try {
        await this.api.removeHistory(original.result.id, signal)
      } catch {
        if (!signal.aborted) this.turn(id, { correctionError: '새 번역은 반영했지만 예전 기록을 지우지 못했습니다. 기록 화면에서 지울 수 있습니다.' })
        return
      }
    } catch (error) {
      if (!signal.aborted) this.turn(id, { correctionError: `방향을 바꾸지 못했습니다. ${errorMessage(error)} 원래 말풍선을 유지했습니다.` })
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
          this.channel?.interrupt()
          this.microphone?.setPaused(true)
          this.update({ playing: true, speaking: false,
            ...(discarded ? { notice: '음성 재생으로 아직 끝나지 않은 말은 보내지 않았습니다. 재생 후 다시 말해 주세요.' } : {}) })
        })
      } catch {
        if (!signal.aborted) this.turn(job.id, { audioError: PLAYBACK_ERROR })
      } finally {
        if (!signal.aborted) {
          this.microphone?.setPaused(Boolean(this.channel && !this.channel.ready))
          this.update({ playing: false, playingId: null })
        }
      }
    }
    if (!signal.aborted) this.playbackPending = false
  }
}

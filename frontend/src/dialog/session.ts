// [파일 역할] 두 사람 대화 세션. 한 화면에서 한국어·영어 화자가 번갈아 말하면, 마디마다 서버가 언어를
// 가려(한국어·영어 중) 상대 언어로 번역·음성을 낸다. 대화 모드(conversation/session.ts)와 같은 마이크·말 끝
// 판정·요청 큐·재생 큐 구조에, 직전 마디 언어 전달과 '방향 바꾸기'가 더해진다.
// [언어 판별] 서버는 판별 확신이 0.6보다 낮으면 번갈아 말한다고 보고 직전 마디의 반대 언어로 처리하고
// language_guessed를 붙인다(docs/experiments.md 10절). 그래서 화면은 직전 마디가 처리된 언어를 알려 주고,
// 추정으로 처리된 마디에는 방향 바꾸기를 제공한다.
// [두 경로] 로그인한 실제 모드는 웹소켓 mode 'dialog'로 192ms 쉼에서 판별·번역을 미리 시작한다. 이때 직전
// 마디 언어는 서버가 연결마다 기억한다. 예시 모드는 1초 판정 뒤 WAV를 POST /api/translate/dialog로 올리며
// previous_lang을 함께 보낸다(docs/api.md).
// [방향 바꾸기] 판별이 틀렸으면 같은 녹음을 반대 방향으로 POST /api/translate/speech에 다시 보내 말풍선을
// 바꾸고, 틀린 기록을 지운다. 그래서 최근 5마디의 녹음만 메모리에 둔다(f3f6afb).
// [연결] DialogPanel.tsx가 만들고 구독한다.

import { errorMessage } from '../api/client'
import type { DialogTranslationResult, Language, TranslationApi, TranslationResult } from '../translation/api'
import { openMicrophone, type Microphone, type OpenMicrophone } from '../conversation/microphone'
import { encodeWav } from '../conversation/pcm'
import { PLAYBACK_ERROR, translationPlayer, type PlayTranslation } from '../conversation/player'
import { createLiveEndpointer, createSileroEndpointer, type CreateEndpointer, type LiveEndpointEvent,
  type SpeechEndpointer } from '../conversation/silero'
import { CanceledUtterance, RECONNECTING, SpeculativeChannel, type Commit } from '../conversation/speculative'
import { isTranslationResult, type OpenSocket } from '../live/socket'

// 말풍선 하나. result는 판별 필드가 붙은 두 사람 대화 결과이거나, 방향을 바꿔 다시 번역한 일반 음성 번역
// 결과다. correctionError는 방향 바꾸기 실패 안내.
export interface DialogTurn {
  id: number
  state: 'queued' | 'translating' | 'done' | 'error' | 'canceled'
  result?: DialogTranslationResult | TranslationResult
  error?: string
  audioError?: string
  correctionError?: string
}
// 화면 전체 상태. correctingId는 방향 바꾸기를 처리 중인 마디(한 번에 하나만).
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
// Job: 요청 큐 항목. utterance는 웹소켓에서 그 마디의 번호(서버에 방향 바꾸기를 알릴 때 쓴다). AudioJob: 재생
// 큐 항목. RECORDING_LIMIT: 방향 바꾸기를 위해 메모리에 둘 최근 녹음 수(5).
type Job = { id: number; utterance?: number; run: (signal: AbortSignal) => Promise<DialogTranslationResult> }
type AudioJob = { id: number; result: TranslationResult }
const RECORDING_LIMIT = 5

// 웹소켓 final의 result가 두 사람 대화 결과인지(음성 번역 결과 + 확신 숫자 + 추정 여부) 확인하는 타입 가드.
function isDialogResult(value: unknown): value is DialogTranslationResult {
  return isTranslationResult(value) && typeof (value as DialogTranslationResult).language_confidence === 'number'
    && typeof (value as DialogTranslationResult).language_guessed === 'boolean'
}

// 그 결과가 '확신이 낮아 번갈아 처리로 정한' 것인지. 방향을 이미 바꾼 결과(일반 음성 번역)에는 필드가 없어
// false다. DialogPanel이 방향 바꾸기 안내를 보일지 정하는 데도 쓴다.
export function wasLanguageGuessed(result: TranslationResult | DialogTranslationResult): result is DialogTranslationResult {
  return 'language_guessed' in result && result.language_guessed
}

// 대화 모드 세션과 같은 필드에 더해:
// - recordings: 마디 번호 → 그 마디의 WAV(최근 5개)
// - previousLang: 직전에 처리된 마디의 원문 언어. HTTP 경로에서 previous_lang으로 보내고, 재연결 때 새 연결의
//   start에 넣는다
// - lastProcessedTurnId·lastProcessedUtterance: 직전에 처리된 마디의 화면 번호와 웹소켓 번호. 방향 바꾸기가
//   '마지막 마디'를 고쳤을 때만 직전 언어를 바꾸기 위해 기억한다
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

  // 시작. 방향을 고르지 않는다(서버가 마디마다 판별). 실제 모드의 start 메시지는 mode 'dialog'이고, 재연결
  // 때는 마지막으로 처리된 마디의 언어와 번호(previous_lang, previous_id)를 함께 넣는다. 나머지 준비
  // 과정(판정기·마이크·연결을 동시에, 처리 못 한 소리 3초 초과 시 멈춤, 실패 단계별 안내)은 대화 모드와 같다.
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

  // 멈춤. 대화 모드처럼 정리하고, 녹음과 직전 마디 정보도 지운다. 다시 시작하면 첫 마디부터 새로 판별한다.
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

  // 판정 이벤트 처리. 대화 모드와 같다: end는 예시 모드면 업로드, 실제 모드면 채널의 확정 결과로 큐에 넣는다.
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
  // 연결이 준비를 잃으면(재연결) 말하던 소리를 버리고 다시 말해 달라고 알린다. 마이크는 준비된 동안에만 연다.
  private connected(ready: boolean) {
    if (!ready) {
      this.captureEpoch++
      this.endpointer?.interrupt()
      this.update({ speaking: false, notice: RECONNECTING })
    }
    this.microphone?.setPaused(!ready || this.state.playing || this.state.permission)
  }

  // 마디를 queued로 추가한다. 방향 바꾸기에 쓰려고 WAV를 만들어 두고, 5개를 넘으면 가장 오래된 녹음부터
  // 지운다(Map은 넣은 순서를 지킨다).
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

  // 요청 큐를 하나씩 처리한다. 결과가 오면 그 마디의 원문 언어를 다음 마디의 직전 언어로 기억한다. HTTP
  // 경로는 요청을 보낼 때 previousLang을 읽으므로, 앞 마디가 끝난 뒤의 값이 들어간다. 실패한 마디는 직전
  // 언어를 바꾸지 않는다.
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

  // '음성 다시 듣기'. 대화 모드와 같다.
  replay(id: number) {
    const turn = this.state.turns.find((entry) => entry.id === id)
    if (!turn?.result || this.state.playingId === id || this.audioJobs.some((entry) => entry.id === id)) return
    this.audioJobs.push({ id, result: turn.result })
    void this.drainAudio()
  }

  // 방향 바꾸기. 추정으로 처리된 마디이고, 녹음이 아직 남아 있고, 다른 바꾸기·재생이 없을 때만 한다.
  // 1) 같은 녹음을 반대 방향(원래 도착 언어 → 원래 출발 언어)으로 POST /api/translate/speech에 보낸다
  // 2) 그 마디가 마지막으로 처리된 마디면 직전 언어를 새 방향으로 바꾸고, 웹소켓이면 서버에도 previous
  //    메시지로 알린다(서버는 그 번호가 마지막 마디일 때만 따른다)
  // 3) 말풍선을 새 결과로 바꾸고 바로 재생한다
  // 4) 틀린 방향의 기록을 지운다. 지우기만 실패하면 새 번역은 그대로 두고 기록 화면에서 지울 수 있다고
  //    알린다(5a6a17c)
  // 다시 번역이 실패하면 원래 말풍선을 그대로 두고 실패를 알린다.
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

  // 그 마디의 녹음이 남아 있는지. 화면이 방향 바꾸기 버튼을 보일지 정하는 데 쓴다(5마디보다 오래된 마디는
  // 바꿀 수 없음).
  hasRecording(id: number) { return this.recordings.has(id) }

  // 재생 큐를 하나씩 재생한다. 대화 모드와 같이, 소리가 나기 시작하면 판정기를 비우고 서버에 진행 중인 마디
  // 취소를 보내고 마이크를 쉬게 하며, 끝나면 다시 듣는다.
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

// [파일 역할] 대화 모드의 두뇌. 버튼 없이 말하고 멈추면 번역 음성까지 나오는 흐름을 React 밖의 상태 객체로
// 관리한다: 마이크 열기 → 말 끝 판정 → 한 마디씩 번역 요청 → 말한 순서대로 결과 표시·재생 → 재생 중에는 듣지
// 않기.
// [두 경로] 로그인한 실제 모드에서는 SpeculativeChannel(웹소켓)로 마디를 흘려보내 서버가 192ms 쉼에서 미리
// 처리하고 1초 판정에서 확정한다. 예시 모드에서는 1초 판정 뒤 마디를 WAV로 만들어 POST
// /api/translate/speech로 올린다. 두 경로 모두 같은 큐(jobs)로 들어가 화면 동작은 같다(00bf7d7,
// docs/experiments.md 11절).
// [설계 이유] 요청과 재생을 각각 한 줄 큐로 돌려, 마디가 겹쳐도 말한 순서대로 보이고 들린다. 번역 음성이
// 나오는 동안 마이크를 쉬게 해 스피커 소리를 다시 말로 듣지 않는다(docs/experiments.md 7절).
// 마이크·판정기·재생기·소켓은 생성자 인자로 바꿔 끼울 수 있어 테스트가 실제 장치 없이 흐름을
// 검사한다(session.test.ts).
// [연결] ConversationPanel.tsx가 만들고 useSyncExternalStore로 상태를 구독한다.

import { errorMessage } from '../api/client'
import type { Direction, TranslationApi, TranslationResult } from '../translation/api'
import { createLiveEndpointer, createSileroEndpointer, type CreateEndpointer, type LiveEndpointEvent,
  type SpeechEndpointer } from './silero'
import { openMicrophone, type Microphone, type OpenMicrophone } from './microphone'
import { encodeWav } from './pcm'
import { PLAYBACK_ERROR, translationPlayer, type PlayTranslation } from './player'
import { CanceledUtterance, RECONNECTING, SpeculativeChannel, type Commit } from './speculative'
import type { OpenSocket } from '../live/socket'

// 화면에 보이는 한 마디. state: queued(차례 기다림) → translating → done 또는 error, 멈추거나 연결이 바뀌어
// 버려지면 canceled. error는 번역 실패 안내, audioError는 재생 실패 안내.
export interface Turn {
  id: number
  state: 'queued' | 'translating' | 'done' | 'error' | 'canceled'
  result?: TranslationResult
  error?: string
  audioError?: string
}
// 화면이 그리는 전체 상태. active: 대화 중, permission: 마이크 권한·판정 파일·연결을 준비하는 중, speaking:
// 말소리를 듣는 중, translating: 번역 요청 처리 중, playing·playingId: 재생 중인 마디, error: 대화를 멈추게
// 한 오류, notice: 짧은 안내.
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
// 요청 큐의 항목: 마디 번호와 결과를 얻는 함수(웹소켓 결과 기다리기 또는 HTTP 업로드).
type Job = { id: number; run: (signal: AbortSignal) => Promise<TranslationResult> }

// 필드 정리.
// - controller: 대화 한 번(start~stop)의 취소 신호. stop마다 새로 만들어, 앞 대화의 늦은 응답이 새 대화를
//   건드리지 못하게 한다
// - captureEpoch: 판정기를 비울 때마다 올리는 번호. 그 전에 넣은 소리에서 나온 이벤트를 버린다
// - direction: 대화를 시작할 때의 번역 방향(시작 뒤 화면에서 바꿔도 이번 대화에는 영향 없음)
// - jobs·requesting: 번역 요청 큐와 처리 중 여부, audioJobs·playbackPending: 재생 큐와 처리 중 여부
export class ConversationSession {
  private state: ConversationState = { active: false, permission: false, speaking: false, translating: false,
    playing: false, playingId: null, turns: [], error: '', notice: '' }
  private listeners = new Set<() => void>()
  private controller = new AbortController()
  private microphone?: Microphone
  private endpointer?: SpeechEndpointer
  // Signed in: the live connection that prepares each clip at a pause (T77). Example mode uploads.
  private channel?: SpeculativeChannel
  private captureEpoch = 0
  private direction: Direction = { source_lang: 'ko', target_lang: 'en' }
  private jobs: Job[] = []
  private audioJobs: Turn[] = []
  private requesting = false
  private playbackPending = false
  private nextId = 1
  private readonly api: Pick<TranslationApi, 'speech' | 'liveToken' | 'expireLiveAuthentication'>
  private readonly open: OpenMicrophone
  private readonly play: PlayTranslation
  private readonly create: CreateEndpointer
  private readonly speculative: boolean
  private readonly socket?: OpenSocket

  // 예시 모드(api.demo)가 아니면 미리 처리(speculative)를 쓰고 판정기도 스트리밍 모드로 만든다. 나머지 인자는
  // 테스트용 교체 지점이다.
  constructor(api: TranslationApi, open: OpenMicrophone = openMicrophone, play: PlayTranslation = translationPlayer(api),
    create?: CreateEndpointer, socket?: OpenSocket) {
    this.api = api; this.open = open; this.play = play; this.socket = socket
    this.speculative = !api.demo
    this.create = create ?? (this.speculative ? createLiveEndpointer : createSileroEndpointer)
  }
  // useSyncExternalStore용 구독 인터페이스(ApiClient와 같은 방식). update는 상태 객체를 통째로 바꿔 React가
  // 변화를 알아차리게 한다.
  getSnapshot = () => this.state
  subscribe = (listener: () => void) => { this.listeners.add(listener); return () => { this.listeners.delete(listener) } }
  private update(patch: Partial<ConversationState>) {
    this.state = { ...this.state, ...patch }
    this.listeners.forEach((listener) => listener())
  }
  private turn(id: number, patch: Partial<Turn>) {
    this.update({ turns: this.state.turns.map((turn) => turn.id === id ? { ...turn, ...patch } : turn) })
  }

  // 대화 시작. 이전 대화를 먼저 정리하고, 실제 모드면 웹소켓 채널(mode 'conversation')을 만든다. 판정기 준비,
  // 마이크 열기, 채널 연결을 동시에 시작해 기다리는 시간을 줄인다.
  async start(direction: Direction) {
    this.stop()
    this.direction = { ...direction }
    const signal = this.controller.signal
    this.update({ active: true, permission: true, error: '', notice: '' })
    const channel = this.speculative ? new SpeculativeChannel(this.api, () => ({ ...direction, mode: 'conversation' }), {
      ready: (ready) => { if (!signal.aborted) this.connected(ready) },
      error: (text) => { if (!signal.aborted) { this.stop(); this.update({ error: text }) } },
    }, '대화 모드', undefined, this.socket) : undefined
    this.channel = channel
    // pendingSamples: 판정기에 넣었지만 아직 처리되지 않은 표본 수. openingMicrophone: 실패가 마이크 단계인지
    // 판정 파일 단계인지 가려 안내 문구를 고르는 데 쓴다.
    let pendingSamples = 0
    let openingMicrophone = true
    try {
      const preparing = this.create()
      // 마이크 조각 처리. 대화 중이 아니거나, 재생 중이거나, 아직 준비 중이거나, 연결이 준비되지 않았으면
      // 버린다.
      const opening = this.open((samples) => {
        if (signal.aborted || !this.state.active || this.state.playing || this.state.permission || !this.endpointer
          || (channel && !channel.ready)) return
        const epoch = this.captureEpoch
        pendingSamples += samples.length
        // 처리 못 한 소리가 3초(16kHz × 3)를 넘으면 기기가 모델 실행을 따라가지 못하는 것이다. 지연이 끝없이
        // 쌓이지 않게 멈추고 알린다.
        if (pendingSamples > 16_000 * 3) {
          this.stop()
          this.update({ error: '말소리 감지가 입력 속도를 따라가지 못해 멈췄습니다. 다른 작업을 줄인 뒤 다시 시작해 주세요.' })
          return
        }
        // 판정 이벤트를 순서대로 처리한다. 그 사이 대화가 멈췄거나 판정기를 비웠으면(epoch가 바뀜) 남은
        // 이벤트는 버린다. 판정 실패는 대화를 멈추는 오류다.
        void this.endpointer.push(samples).then((events) => {
          for (const event of events) {
            if (signal.aborted || epoch !== this.captureEpoch) return
            this.event(event)
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
      // 마이크가 열렸을 때 이미 멈췄으면 바로 닫는다. 실제 모드면 연결이 준비될 때까지 마이크를 쉬게 둔다.
      }).then((microphone) => {
        if (signal.aborted) microphone.stop()
        else { this.microphone = microphone; if (channel) microphone.setPaused(true) }
        openingMicrophone = false
      })
      // 셋이 모두 준비되면 판정기를 붙이고 '준비 중'을 끝낸다. 연결이 준비됐고 재생 중이 아니면 마이크를 다시
      // 연다.
      const [endpointer] = await Promise.all([preparing, opening, channel?.start(signal)])
      if (signal.aborted) { endpointer.interrupt(); return }
      this.endpointer = endpointer
      this.update({ permission: false })
      if (channel) this.microphone?.setPaused(!channel.ready || this.state.playing)
    } catch {
      // 사용자가 멈춘 것이면 안내하지 않는다. 그 밖에는 실패 단계에 맞는 안내를 띄운다.
      if (signal.aborted) return
      this.stop()
      this.update({ error: openingMicrophone
        ? '대화를 준비하지 못했습니다. 마이크 권한과 HTTPS 연결, 브라우저 지원을 확인해 주세요.'
        : '말소리 감지 파일을 준비하지 못했습니다. 연결을 확인하고 다시 시작해 주세요.' })
    }
  }

  // 대화 멈춤. 취소 신호를 보내 진행 중인 요청·재생을 끝내고, 채널·마이크·판정기를 닫고, 큐를 비운다. 아직
  // 끝나지 않은 마디는 canceled로 표시하고 이미 끝난 마디는 그대로 둔다. 화살표 함수라 컴포넌트 정리 함수에
  // 그대로 넘길 수 있다.
  stop = () => {
    this.controller.abort()
    this.controller = new AbortController()
    this.channel?.stop(); this.channel = undefined
    this.microphone?.stop(); this.microphone = undefined
    this.captureEpoch++
    this.endpointer?.interrupt(); this.endpointer = undefined
    this.jobs = []; this.audioJobs = []
    this.requesting = this.playbackPending = false
    this.update({ active: false, permission: false, speaking: false, translating: false, playing: false, playingId: null,
      notice: '', turns: this.state.turns.map((turn) => turn.state === 'queued' || turn.state === 'translating'
        ? { ...turn, state: 'canceled' } : turn) })
  }

  // 판정 이벤트 처리. 소리·pause·resume은 채널로 그대로 넘기고(예시 모드면 채널이 없어 무시),
  // start·end·discard는 화면 상태도 바꾼다. end는 예시 모드면 WAV 업로드로, 실제 모드면 채널의 확정 결과로
  // 큐에 넣는다. 너무 짧은 소리는 건너뛰었다고 알린다.
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
  // 연결이 준비를 잃으면(토큰 만료로 재연결) 듣던 말을 버리고 다시 말해 달라고 알린다. 서버가 그 마디를 이미
  // 잃었기 때문이다. 마이크는 준비된 동안에만 연다.
  private connected(ready: boolean) {
    if (!ready) {
      this.captureEpoch++
      this.endpointer?.interrupt()
      this.update({ speaking: false, notice: RECONNECTING })
    }
    this.microphone?.setPaused(!ready || this.state.playing || this.state.permission)
  }

  // 마디를 화면에 queued로 추가하고 요청 큐에 넣는다. 업로드 경로는 지금 방향과 WAV를 이때 고정해 둔다.
  private enqueue(samples: Float32Array, commit?: Commit<TranslationResult>) {
    const id = this.nextId++
    this.update({ turns: [...this.state.turns, { id, state: 'queued' }] })
    if (commit) this.jobs.push({ id, run: () => commit.result })
    else {
      const audio = encodeWav(samples), direction = this.direction
      this.jobs.push({ id, run: (signal) => this.api.speech(audio, direction, signal) })
    }
    void this.drainRequests()
  }

  // 요청 큐를 하나씩 처리하는 루프(동시에 하나만 돈다). 결과는 말한 순서대로 화면에 붙고 재생 큐로 넘어간다.
  // 한 마디가 실패해도 그 마디에만 오류를 표시하고 다음 마디를 계속 처리한다. 연결이 바뀌어 결과가 오지 않을
  // 마디(CanceledUtterance)는 오류 대신 canceled로 둔다.
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
        this.turn(job.id, { state: 'done', result })
        this.audioJobs.push({ id: job.id, state: 'done', result })
        void this.drainAudio()
      } catch (error) {
        if (signal.aborted) return
        this.turn(job.id, error instanceof CanceledUtterance ? { state: 'canceled' } : { state: 'error', error: errorMessage(error) })
      }
    }
    if (!signal.aborted) { this.requesting = false; this.update({ translating: false }) }
  }

  // '음성 다시 듣기'. 결과가 있고 지금 재생 중이거나 이미 재생 큐에 있지 않을 때만 다시 넣는다.
  replay(id: number) {
    const turn = this.state.turns.find((entry) => entry.id === id)
    if (!turn?.result || this.state.playingId === id || this.audioJobs.some((entry) => entry.id === id)) return
    this.audioJobs.push(turn)
    void this.drainAudio()
  }

  // 재생 큐를 하나씩 재생하는 루프. 실제로 소리가 나기 시작하면(playing 콜백) 판정기를 비우고, 채널에 진행
  // 중인 마디 취소를 보내고, 마이크를 쉬게 한다. 말하던 중이었다면 그 말은 보내지 않았다고 알린다. 재생이
  // 끝나면 연결이 준비된 경우에만 마이크를 다시 연다.
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
          this.channel?.interrupt()
          this.microphone?.setPaused(true)
          this.update({ playing: true, speaking: false,
            ...(discarded ? { notice: '음성 재생으로 아직 끝나지 않은 말은 보내지 않았습니다. 재생 후 다시 말해 주세요.' } : {}) })
        })
      } catch {
        if (signal.aborted) return
        this.turn(turn.id, { audioError: PLAYBACK_ERROR })
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

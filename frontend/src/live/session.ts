// [파일 역할] 동시통역 세션. 말하는 동안 원문·번역문 자막을 계속 갱신해 보여 주고, 말을 멈추면 대화 모드와
// 같은 최종 결과와 번역 음성을 낸다. React 밖의 상태 객체로 마이크, 말 끝 판정(스트리밍 모드), 웹소켓, 재생을
// 묶는다.
// [흐름] 시작 → 판정 모델 준비·마이크 열기·웹소켓 연결을 동시에 → 마디가 시작되면 utterance와 소리를 보냄 →
// 서버가 약 1초마다 그 마디 전체를 다시 인식·번역해 source·translation 자막을 보냄 → 192ms 쉼에 pause(서버가
// 최종 처리를 미리 시작) → 1초 쉼에 end → final이 오면 결과 표시·음성 재생(재생 중에는 듣지 않음).
// [설계 이유] 자막 갱신 간격(1초)과 진하게 보일 앞부분(stable)은 서버가 정하고 화면은 그대로 그린다. 번역문은
// 앞부분도 자주 바뀌어 서버가 stable을 늘 0으로 보내므로 최종 결과 전까지 흐리게 보인다(docs/api.md,
// docs/experiments.md 8절).
// [연결] LivePanel.tsx가 만들고 구독한다. 연결 관리는 socket.ts의 LiveTransport, 소리 처리는 conversation/
// 폴더의 마이크·판정기·PCM 변환·재생기를 함께 쓴다.

import type { Direction, TranslationApi, TranslationResult } from '../translation/api'
import { openMicrophone, type Microphone, type OpenMicrophone } from '../conversation/microphone'
import { createLiveEndpointer, type CreateEndpointer, type SpeechEndpointer, type LiveEndpointEvent } from '../conversation/silero'
import { encodePcm16 } from '../conversation/pcm'
import { PLAYBACK_ERROR, translationPlayer, type PlayTranslation } from '../conversation/player'
import { isTranslationResult, LiveTransport, utteranceError, type OpenSocket } from './socket'

// 자막 한 줄. stable은 앞에서부터 진하게(확정) 보일 글자 수(유니코드 코드 포인트 단위)다.
export interface Caption { text: string; stable: number }
// 화면의 한 마디. state: speaking(말하는 중) → waiting(end를 보내고 최종 결과 기다림) → done 또는 error, 짧은
// 소리·재생·재연결·멈춤으로 버려지면 canceled. direction은 그 마디를 말할 때의 방향.
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
// 화면 전체 상태. ready: 웹소켓이 서버의 ready를 받은 상태. announcement: 최종 결과가 나올 때 스크린리더에 한
// 번 읽어 줄 문장(자막이 계속 바뀌는 동안에는 읽지 않게 따로 둔다).
export interface LiveState {
  active: boolean; permission: boolean; ready: boolean; speaking: boolean; playing: boolean
  playingId: number | null; turns: LiveTurn[]; error: string; notice: string; announcement: string
}
// 필드: controller는 시작~멈춤 한 번의 취소 신호, captureEpoch는 판정기를 비울 때마다 올려 그 전 소리의
// 이벤트를 버리는 번호, current는 지금 말하는 마디 번호, nextId는 다음 마디 번호, audioJobs·playbackPending은
// 재생 큐. 마이크·재생·판정기·소켓은 생성자 인자로 바꿔 끼울 수 있다(session.test.ts).
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

  // useSyncExternalStore용 구독 인터페이스. fail은 멈춘 뒤 오류 문구를 띄운다.
  getSnapshot = () => this.state
  subscribe = (listener: () => void) => { this.listeners.add(listener); return () => { this.listeners.delete(listener) } }
  private update(patch: Partial<LiveState>) {
    this.state = { ...this.state, ...patch }; this.listeners.forEach((listener) => listener())
  }
  private turn(id: number, patch: Partial<LiveTurn>) {
    this.update({ turns: this.state.turns.map((turn) => turn.id === id ? { ...turn, ...patch } : turn) })
  }
  private fail = (error: string) => { this.stop(); this.update({ error }) }

  // 시작. 앞 세션을 정리하고 웹소켓 전송 층을 만든다. start 메시지에는 방향만 넣는다(mode가 없으면 서버는
  // 동시통역으로 본다, docs/api.md).
  async start(direction: Direction) {
    this.stop()
    this.direction = { ...direction }
    const signal = this.controller.signal
    this.update({ active: true, permission: true, error: '', notice: '' })
    const transport = new LiveTransport(this.api, () => ({ ...direction }), {
      // 연결 준비 상태가 바뀔 때: 준비를 잃었으면(토큰 만료로 재연결) 말하던 마디와 결과를 기다리던 마디를
      // 모두 버리고 다시 말해 달라고 알린다. 새 연결의 서버는 옛 마디를 모르기 때문이다. 마이크는 연결이
      // 준비되고 재생·준비 중이 아닐 때만 연다.
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
        // 마이크 조각 처리: 준비 중, 연결 미준비, 재생 중이면 버린다. 처리 못 한 소리가 48,000표본(16kHz로
        // 3초)을 넘으면 기기가 판정을 따라가지 못하는 것이므로 멈춘다. 판정 이벤트는 순서대로 처리하되, 그
        // 사이 판정기를 비웠으면(epoch 변경) 남은 이벤트를 버린다.
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
        // 마이크가 열려도 연결이 준비될 때까지는 쉬게 둔다.
        }).then((microphone) => {
          if (signal.aborted) microphone.stop()
          else { this.microphone = microphone; microphone.setPaused(true) }
        }),
        transport.start(signal),
      ])
      // 세 가지가 모두 준비되면 판정기를 붙이고 준비 중 표시를 끈 뒤, 연결이 준비됐고 재생 중이 아니면 듣기
      // 시작한다.
      if (signal.aborted) return
      this.endpointer = endpointer
      this.update({ permission: false })
      this.microphone?.setPaused(!this.state.ready || this.state.playing)
    } catch {
      if (!signal.aborted) this.fail('동시통역을 준비하지 못했습니다. 마이크 권한과 HTTPS 연결을 확인해 주세요.')
    }
  }

  // 판정 이벤트를 웹소켓 메시지와 화면 상태로 바꾼다. start: 새 마디를 화면에 추가하고 utterance 전송. audio:
  // 16비트 PCM 바이너리. pause·resume: 그대로 전송. end: 마디를 waiting으로 바꾸고 end 전송. discard: 그
  // 마디를 취소(cancel)하고 짧은 소리라 건너뛰었다고 알린다.
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

  // 서버 메시지 처리. 이미 끝났거나 버린 마디에 대한 메시지는 무시한다.
  // - source·translation: 자막 갱신. stable은 정수인지 확인하고 0~글자 수 범위로 잘라, 잘못된 값이 와도
  //   화면이 깨지지 않게 한다
  // - error: 그 마디만 실패로 표시(연결은 유지)
  // - final: 최종 결과로 자막을 모두 진하게 바꾸고, 스크린리더 안내 문장을 만들고, 재생 큐에 넣는다
  private receive(value: unknown) {
    if (!value || typeof value !== 'object' || !('id' in value) || !('type' in value)) return
    const turn = this.state.turns.find((entry) => entry.id === value.id)
    if (!turn || turn.state === 'done' || turn.state === 'error' || turn.state === 'canceled') return
    if ((value.type === 'source' || value.type === 'translation') && 'text' in value && 'stable' in value
      && typeof value.text === 'string' && typeof value.stable === 'number' && Number.isInteger(value.stable)) {
      const length = Array.from(value.text).length // server counts Unicode code points, not UTF-16 units
      this.turn(turn.id, { [value.type]: { text: value.text, stable: Math.max(0, Math.min(length, value.stable)) } })
    } else if (value.type === 'error') {
      this.turn(turn.id, { state: 'error', error: utteranceError('detail' in value ? value.detail : undefined) })
    } else if (value.type === 'final' && 'result' in value && isTranslationResult(value.result)) {
      const result = value.result
      this.turn(turn.id, { state: 'done', result, source: { text: result.source_text, stable: Array.from(result.source_text).length },
        translation: { text: result.translated_text, stable: Array.from(result.translated_text).length } })
      this.update({ announcement: `${turn.id}번째 번역 완료. 원문: ${result.source_text} 번역문: ${result.translated_text}` })
      this.audioJobs.push({ ...turn, state: 'done', result })
      void this.drainAudio()
    }
  }

  // 말하던 마디를 canceled로 바꾸고 서버에 cancel을 보낸다.
  private cancelCurrent() {
    if (this.current !== null) {
      const id = this.current
      this.current = null
      this.turn(id, { state: 'canceled' })
      this.transport?.send({ type: 'cancel', id })
    }
    this.update({ speaking: false })
  }
  // 판정기를 비우고(epoch 증가로 처리 중이던 이벤트도 버림) 말하던 마디를 취소한다. 재생 시작, 재연결, 멈춤
  // 때 부른다.
  private interrupt() {
    this.captureEpoch++
    this.endpointer?.interrupt()
    this.cancelCurrent()
  }
  // 말하는 중이거나 결과를 기다리던 마디를 모두 canceled로 표시한다.
  private cancelPending() {
    this.update({ turns: this.state.turns.map((turn) => turn.state === 'waiting' || turn.state === 'speaking'
      ? { ...turn, state: 'canceled' } : turn) })
  }
  // 멈춤: 말하던 마디와 결과를 기다리던 마디에 cancel을 보내 서버가 하던 처리를 버리게 한 뒤, 취소 신호를
  // 보내고 소켓·마이크를 닫고 큐를 비운다. 이미 끝난 마디는 화면에 남긴다.
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
  // '음성 다시 듣기'. 결과가 있고 지금 재생 중이거나 큐에 있지 않을 때만 넣는다.
  replay(id: number) {
    const turn = this.state.turns.find((entry) => entry.id === id)
    if (!turn?.result || this.state.playingId === id || this.audioJobs.some((entry) => entry.id === id)) return
    this.audioJobs.push(turn)
    void this.drainAudio()
  }
  // 재생 큐를 하나씩 재생한다. 소리가 나기 시작하면 판정기를 비우고 말하던 마디를 취소하고 마이크를 쉬게
  // 한다(스피커 소리를 다시 인식하지 않게). 재생이 끝나면 연결이 준비된 경우에만 다시 듣는다. 재생 실패는 그
  // 마디에만 표시한다.
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

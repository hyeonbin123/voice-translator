// [파일 역할] 번역 관련 서버 API(글 번역, 음성 번역, 두 사람 대화, 기록 삭제, 번역 음성 받기)를 부르는 얇은
// 층과 그 응답 타입. 웹소켓 기능이 쓸 토큰 받기·만료 처리도 여기서 ApiClient로 넘긴다.
// [설계 이유] 화면과 대화 세션 코드는 이 클래스만 알고 fetch·토큰·오류 문구는 몰라도 된다. 생성자가 ApiClient
// 전체가 아니라 request(와 선택적으로 liveToken·logout)만 요구하므로, 예시 모드(mock.ts)와 테스트가 request
// 하나만 가진 가짜 객체를 넣어 같은 화면을 돌릴 수 있다.
// [연결] App.tsx가 실제 api 클라이언트로 만들거나 예시 모드 객체를 골라 번역 화면에 넘긴다. history/api.ts의
// HistoryApi가 이 클래스를 이어받는다. 요청·응답 모양은 docs/api.md 번역 절을 따른다.

import type { ApiClient } from '../api/client'

// 지원 언어는 한국어와 영어 둘뿐이다. Direction은 출발·도착 언어 쌍이며 서버는 둘이 같으면 422로 거절한다.
export type Language = 'ko' | 'en'
export interface Direction { source_lang: Language; target_lang: Language }
// 번역 응답(201) 한 건 = 저장된 기록 한 건 + tts_error. 시간 필드(*_ms)는 단계별 모델 호출 시간(밀리초)이고
// 실행하지 않은 인식·합성은 null이다. audio_id가 null이면 서버 음성이 없어 화면이 브라우저 내장 음성으로
// 읽는다. tts_error는 합성 실패 이유를 담는 고정 문자열 두 가지 중 하나다(docs/api.md).
export interface TranslationResult extends Direction {
  id: string
  mode: 'text' | 'speech'
  source_text: string
  translated_text: string
  stt_model: string | null
  mt_model: string
  tts_model: string | null
  stt_ms: number | null
  mt_ms: number
  tts_ms: number | null
  audio_id: string | null
  created_at: string
  tts_error: 'Speech synthesis failed' | 'Speech synthesis is not available' | null
}
// 두 사람 대화 응답에 붙는 두 필드: 판별한 언어의 확신(0.5~1.0)과, 확신이 기준(0.6)보다 낮아 '번갈아
// 말한다'고 보고 방향을 정했는지(language_guessed). 화면은 language_guessed가 true일 때 방향 바꾸기를 보여
// 준다(docs/experiments.md 10절).
export interface DialogTranslationResult extends TranslationResult {
  language_confidence: number
  language_guessed: boolean
}

// 업로드 크기 한도. 서버는 10MiB(10×1024×1024바이트)까지 받지만 화면은 그보다 조금 작은 10,000,000바이트로
// 미리 막아, 한도 근처 파일이 올라가서 413이 되는 일을 피한다. 30초 길이 제한은 서버가 디코딩해서 확인한다.
// 쓰는 곳: TranslatePage.tsx(파일 선택), useRecorder.ts(녹음 중 크기).
export const MAX_AUDIO_BYTES = 10_000_000 // Conservative decimal MB; server validates decoded duration.
// 글자 수 세기. 서버 규칙(앞뒤 공백을 자른 뒤 유니코드 문자 1~500자)과 같게 trim한 뒤 코드 포인트 단위로
// 센다. 번역 화면의 글자 수 표시와 500자 검사에 쓴다.
export const characterCount = (text: string) => Array.from(text.trim()).length

// client: 인증 요청을 보낼 객체. demo: 예시 모드 여부. 예시 모드에서는 서버 웹소켓을 쓸 수 없으므로 동시통역
// 버튼을 끄고(LivePanel), 대화 모드·두 사람 대화는 미리 처리 대신 말이 끝난 뒤 HTTP로
// 올린다(conversation/session.ts, dialog/session.ts의 speculative).
export class TranslationApi {
  private readonly client: Pick<ApiClient, 'request'> & Partial<Pick<ApiClient, 'liveToken' | 'logout'>>
  readonly demo: boolean

  constructor(client: Pick<ApiClient, 'request'> & Partial<Pick<ApiClient, 'liveToken' | 'logout'>>, demo = false) {
    this.client = client
    this.demo = demo
  }

  // 웹소켓 첫 메시지용 access 토큰. 예시 모드이거나 클라이언트에 토큰 기능이 없으면 바로 실패한다. 실제
  // 처리는 ApiClient.liveToken이 한다.
  async liveToken(rejectedToken?: string): Promise<string> {
    if (this.demo || !this.client.liveToken) throw new Error('Live translation needs authentication')
    return this.client.liveToken(rejectedToken)
  }

  // 웹소켓이 4401(토큰 거절)로 닫혔고 다시 받은 토큰으로도 안 될 때 live/socket.ts가 부른다. 만료
  // 로그아웃으로 로그인 화면에 '만료' 안내를 띄운다.
  expireLiveAuthentication() { this.client.logout?.(true) }

  // 글 번역: POST /api/translate/text에 JSON으로 보낸다. 앞뒤 공백은 화면에서도 잘라 보낸다. signal로 화면을
  // 떠나거나 새 요청을 보낼 때 이전 요청을 취소한다.
  async text(text: string, direction: Direction, signal: AbortSignal): Promise<TranslationResult> {
    return (await this.client.request('/api/translate/text', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text.trim(), ...direction }), signal,
    })).json()
  }

  // 음성 번역: POST /api/translate/speech에 multipart로 녹음과 언어를 보낸다. 서버는 파일의 Content-Type을
  // 믿지 않고 실제로 디코딩해 보므로(docs/api.md), 파일 이름은 MIME에 맞춘 확장자를 붙인 참고용 이름이다.
  // 사용자가 고른 파일(File)은 원래 이름을 쓴다. 대화 모드와 두 사람 대화의 방향 바꾸기도 이 함수로 WAV를
  // 보낸다.
  async speech(audio: Blob, direction: Direction, signal: AbortSignal): Promise<TranslationResult> {
    const body = new FormData()
    const extension = audio.type.includes('ogg') ? 'ogg' : audio.type.includes('mp4') ? 'm4a' : 'webm'
    body.append('audio', audio, audio instanceof File ? audio.name : `recording.${extension}`)
    body.append('source_lang', direction.source_lang)
    body.append('target_lang', direction.target_lang)
    return (await this.client.request('/api/translate/speech', { method: 'POST', body, signal })).json()
  }

  // 두 사람 대화(HTTP 방식): POST /api/translate/dialog. 언어 대신 직전 마디가 처리된 언어를 보내고, 첫
  // 마디는 보내지 않는다. 서버가 언어를 판별해 반대 언어로 번역한다.
  async dialog(audio: Blob, previousLang: Language | undefined, signal: AbortSignal): Promise<DialogTranslationResult> {
    const body = new FormData()
    body.append('audio', audio, audio instanceof File ? audio.name : 'conversation.wav')
    if (previousLang) body.append('previous_lang', previousLang)
    return (await this.client.request('/api/translate/dialog', { method: 'POST', body, signal })).json()
  }

  // 기록 삭제. 두 사람 대화에서 방향을 바꿔 다시 번역했을 때 틀린 방향의 기록을 지우는 데
  // 쓴다(dialog/session.ts). id는 주소에 넣으므로 encodeURIComponent로 감싼다.
  async removeHistory(id: string, signal: AbortSignal): Promise<void> {
    await this.client.request(`/api/history/${encodeURIComponent(id)}`, { method: 'DELETE', signal })
  }

  // 번역 음성(WAV) 받기. <audio src>에는 인증 머리글을 붙일 수 없어 fetch로 받은 Blob을 화면이 blob URL로
  // 바꿔 재생한다(docs/api.md GET /api/audio/{id}). 쓰는 곳: translation/Playback.tsx,
  // conversation/player.ts.
  async audio(id: string, signal: AbortSignal): Promise<Blob> {
    return (await this.client.request(`/api/audio/${encodeURIComponent(id)}`, { signal })).blob()
  }
}

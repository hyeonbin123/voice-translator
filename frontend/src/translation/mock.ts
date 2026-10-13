// [파일 역할] 예시 모드용 가짜 번역 API. VITE_TRANSLATION_MOCK=true로 개발 서버를 띄우면 App.tsx가 실제 API
// 대신 이것을 번역 화면에 넘긴다. 모델 없이(LOAD_MODELS 없이) 서버를 띄운 상태에서도 화면 흐름을 볼 수 있게
// 하려는 것이다(README 로컬 개발).
// [설계 이유] TranslationApi에 request 하나만 가진 객체를 넣어, 화면 코드는 실제 모드와 똑같이 움직이고
// 응답만 고정 예시가 된다. 응답 모양은 docs/api.md의 201 응답과 같게 맞춰 계약 확인용으로도 쓴다. 마지막 인자
// true(demo)라 웹소켓 기능은 꺼진다.

import { ApiError } from '../api/client'
import { TranslationApi, type TranslationResult, type Direction } from './api'

// Development-only contract fixture. No requests, recordings or history are stored.
export const mockTranslationApi = new TranslationApi({
  // 가짜 request: 실제 네트워크 요청 없이 Response 객체를 만들어 돌려준다. 취소된 요청은 실제 fetch처럼
  // AbortError로 끝낸다. 기록 삭제(DELETE)는 언제나 204로 성공한다.
  async request(path, init) {
    init?.signal?.throwIfAborted()
    if (init?.method === 'DELETE') return new Response(null, { status: 204 })
    const speech = path === '/api/translate/speech'
    const dialog = path === '/api/translate/dialog'
    const recorded = speech || dialog
    const form = init?.body as FormData
    const previous = dialog ? form.get('previous_lang') : null
    // 요청에서 방향과 글을 꺼낸다. 음성 번역은 폼의 언어 필드를, 글 번역은 JSON 본문을 쓴다. 두 사람 대화는
    // 실제 서버처럼 직전 마디의 반대 언어로 처리한 것처럼 꾸민다(직전이 ko면 영어 → 한국어, 아니면 한국어 →
    // 영어).
    const input: Direction & { text: string } = speech || dialog
      ? { source_lang: dialog ? (previous === 'ko' ? 'en' : 'ko') : form.get('source_lang'),
        target_lang: dialog ? (previous === 'ko' ? 'ko' : 'en') : form.get('target_lang'), text: '' }
      : JSON.parse(init?.body as string)
    // 번역할 수 있는 예시 문장은 두 개뿐이다. 녹음은 내용을 알 수 없으므로 출발 언어의 첫 예시('안녕하세요'
    // 또는 'Hello')를 인식한 것으로 친다. 목록에 없는 글은 422 안내로 거절한다.
    const samples = new Map(input.source_lang === 'ko'
      ? [['안녕하세요', 'Hello'], ['감사합니다', 'Thank you']]
      : [['Hello', '안녕하세요'], ['Thank you', '감사합니다']])
    const source = recorded ? (input.source_lang === 'ko' ? '안녕하세요' : 'Hello') : input.text
    const translated = samples.get(source)
    if (!translated) throw new ApiError(422, '예시 모드에서는 안내된 예시 문장을 입력해 주세요.')
    // 고정 응답. 서버 음성은 없다고 답해(audio_id null, tts_error 'not available') 화면이 브라우저 내장
    // 음성으로 읽는 경로를 탄다.
    const body: TranslationResult = {
      id: '080b161c-53d6-4460-992b-f778a5e348cd', mode: recorded ? 'speech' : 'text',
      source_lang: input.source_lang, target_lang: input.target_lang,
      source_text: source, translated_text: translated,
      stt_model: recorded ? 'example-stt' : null, mt_model: 'example-mt', tts_model: null,
      stt_ms: recorded ? 200 : null, mt_ms: 120, tts_ms: null, audio_id: null,
      created_at: new Date().toISOString(), tts_error: 'Speech synthesis is not available',
    }
    // 두 사람 대화의 두 번째 마디부터는 확신 0.62로 '추정' 표시(language_guessed)를 붙여 방향 바꾸기 안내를
    // 화면에서 볼 수 있게 한다.
    if (dialog) Object.assign(body, { language_confidence: previous ? .62 : .98, language_guessed: Boolean(previous) })
    return new Response(JSON.stringify(body), { status: 201, headers: {
      'Content-Type': 'application/json', Location: `/api/history/${body.id}`,
    } })
  },
}, true)

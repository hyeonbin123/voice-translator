// [파일 역할] 예시 모드 가짜 API(mock.ts) 테스트. 서버 없이 화면을 볼 때 쓰는 응답이 실제 API
// 계약(docs/api.md)의 모양을 지키는지 확인한다. 예시 응답이 실제와 다르면 예시 모드에서만 보이는 화면 오류가
// 생기기 때문이다.

import { expect, it } from 'vitest'
import { ApiError } from '../api/client'
import { mockTranslationApi } from './mock'

const wav = () => new File([new Uint8Array(44)], 'conversation.wav', { type: 'audio/wav' })
const signal = () => new AbortController().signal

// 확인: 두 사람 대화의 첫 마디(직전 언어 없음)는 한국어 예시를 확신 0.98, 추정 아님으로 돌려준다.
it('answers a first two-person turn with the fixed Korean sample as a confident speech result', async () => {
  const result = await mockTranslationApi.dialog(wav(), undefined, signal())
  expect(result).toMatchObject({ mode: 'speech', source_lang: 'ko', target_lang: 'en', source_text: '안녕하세요',
    translated_text: 'Hello', stt_model: 'example-stt', stt_ms: 200, language_confidence: .98, language_guessed: false })
})

// 확인: 직전이 한국어면 다음 마디를 영어로 처리하고 추정(language_guessed)으로 표시해, 예시 모드에서도 방향
// 바꾸기 화면을 볼 수 있다.
it('switches a two-person turn after Korean to English and marks the direction as guessed', async () => {
  const result = await mockTranslationApi.dialog(wav(), 'ko', signal())
  expect(result).toMatchObject({ mode: 'speech', source_lang: 'en', target_lang: 'ko', source_text: 'Hello',
    translated_text: '안녕하세요', language_confidence: .62, language_guessed: true })
})

// 확인: 음성 번역은 예시 인식 결과를, 글 번역은 예시 문장의 번역을 돌려주며 글 번역에는 인식 필드가 null이고
// 판별 필드가 없다. 목록에 없는 글은 422 ApiError로 거절한다.
it('keeps the speech and typed text samples', async () => {
  expect(await mockTranslationApi.speech(new Blob(['audio']), { source_lang: 'ko', target_lang: 'en' }, signal()))
    .toMatchObject({ mode: 'speech', source_text: '안녕하세요', translated_text: 'Hello', stt_model: 'example-stt' })
  const text = await mockTranslationApi.text('Hello', { source_lang: 'en', target_lang: 'ko' }, signal())
  expect(text).toMatchObject({ mode: 'text', translated_text: '안녕하세요', stt_model: null, stt_ms: null })
  expect(text).not.toHaveProperty('language_confidence')
  const unknown = mockTranslationApi.text('Good morning', { source_lang: 'en', target_lang: 'ko' }, signal())
  await expect(unknown).rejects.toBeInstanceOf(ApiError)
  await expect(unknown).rejects.toMatchObject({ status: 422 })
})

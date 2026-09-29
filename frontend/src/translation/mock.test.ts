import { expect, it } from 'vitest'
import { ApiError } from '../api/client'
import { mockTranslationApi } from './mock'

const wav = () => new File([new Uint8Array(44)], 'conversation.wav', { type: 'audio/wav' })
const signal = () => new AbortController().signal

it('answers a first two-person turn with the fixed Korean sample as a confident speech result', async () => {
  const result = await mockTranslationApi.dialog(wav(), undefined, signal())
  expect(result).toMatchObject({ mode: 'speech', source_lang: 'ko', target_lang: 'en', source_text: '안녕하세요',
    translated_text: 'Hello', stt_model: 'example-stt', stt_ms: 200, language_confidence: .98, language_guessed: false })
})

it('switches a two-person turn after Korean to English and marks the direction as guessed', async () => {
  const result = await mockTranslationApi.dialog(wav(), 'ko', signal())
  expect(result).toMatchObject({ mode: 'speech', source_lang: 'en', target_lang: 'ko', source_text: 'Hello',
    translated_text: '안녕하세요', language_confidence: .62, language_guessed: true })
})

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

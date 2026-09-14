import { ApiError } from '../api/client'
import { TranslationApi, type TranslationResult, type Direction } from './api'

// Development-only contract fixture. No requests, recordings or history are stored.
export const mockTranslationApi = new TranslationApi({
  async request(path, init) {
    init?.signal?.throwIfAborted()
    if (init?.method === 'DELETE') return new Response(null, { status: 204 })
    const speech = path === '/api/translate/speech'
    const dialog = path === '/api/translate/dialog'
    const form = init?.body as FormData
    const previous = dialog ? form.get('previous_lang') : null
    const input: Direction & { text: string } = speech || dialog
      ? { source_lang: dialog ? (previous === 'ko' ? 'en' : 'ko') : form.get('source_lang'),
        target_lang: dialog ? (previous === 'ko' ? 'ko' : 'en') : form.get('target_lang'), text: '' }
      : JSON.parse(init?.body as string)
    const samples = new Map(input.source_lang === 'ko'
      ? [['안녕하세요', 'Hello'], ['감사합니다', 'Thank you']]
      : [['Hello', '안녕하세요'], ['Thank you', '감사합니다']])
    const source = speech ? (input.source_lang === 'ko' ? '안녕하세요' : 'Hello') : input.text
    const translated = samples.get(source)
    if (!translated) throw new ApiError(422, '예시 모드에서는 안내된 예시 문장을 입력해 주세요.')
    const body: TranslationResult = {
      id: '080b161c-53d6-4460-992b-f778a5e348cd', mode: speech ? 'speech' : 'text',
      source_lang: input.source_lang, target_lang: input.target_lang,
      source_text: source, translated_text: translated,
      stt_model: speech ? 'example-stt' : null, mt_model: 'example-mt', tts_model: null,
      stt_ms: speech ? 200 : null, mt_ms: 120, tts_ms: null, audio_id: null,
      created_at: new Date().toISOString(), tts_error: 'Speech synthesis is not available',
    }
    if (dialog) Object.assign(body, { language_confidence: previous ? .62 : .98, language_guessed: Boolean(previous) })
    return new Response(JSON.stringify(body), { status: 201, headers: {
      'Content-Type': 'application/json', Location: `/api/history/${body.id}`,
    } })
  },
}, true)

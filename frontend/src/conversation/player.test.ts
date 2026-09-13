import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { translationPlayer } from './player'
import type { TranslationResult } from '../translation/api'

const result: TranslationResult = { id: 'turn', mode: 'speech', source_lang: 'ko', target_lang: 'en',
  source_text: '안녕', translated_text: 'Hello', stt_model: 'stt', mt_model: 'mt', tts_model: 'tts',
  stt_ms: 1, mt_ms: 1, tts_ms: 1, audio_id: 'sound', tts_error: null, created_at: '' }
const pause = vi.fn(), play = vi.fn(), removeAttribute = vi.fn(), revoke = vi.fn(), createUrl = vi.fn()
let elements: { onended: (() => void) | null; onerror: (() => void) | null }[] = []
beforeEach(() => {
  elements = []
  vi.clearAllMocks(); play.mockResolvedValue(undefined); createUrl.mockReturnValue('blob:test')
  vi.stubGlobal('Audio', class {
    onended: (() => void) | null = null; onerror: (() => void) | null = null
    pause = pause; play = play; removeAttribute = removeAttribute
    constructor() { elements.push(this) }
  })
  vi.stubGlobal('URL', class extends URL {
    static createObjectURL = createUrl
    static revokeObjectURL = revoke
  })
})
afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks() })

it.each(['end', 'abort', 'error'])('fetches authenticated audio and cleans up on %s', async (ending) => {
  const audio = vi.fn().mockResolvedValue(new Blob(['wav']))
  const controller = new AbortController(), playing = vi.fn()
  const task = translationPlayer({ audio })(result, controller.signal, playing)
  const outcome = task.catch((error) => error)
  await vi.waitFor(() => expect(play).toHaveBeenCalledOnce())
  expect(audio).toHaveBeenCalledWith('sound', controller.signal)
  expect(playing).toHaveBeenCalledOnce()
  if (ending === 'abort') controller.abort()
  else if (ending === 'error') elements[0].onerror!()
  else elements[0].onended!()
  const value = await outcome
  expect(value instanceof Error).toBe(ending === 'error')
  expect(pause).toHaveBeenCalledOnce()
  expect(revoke).toHaveBeenCalledWith('blob:test')
  expect(removeAttribute).toHaveBeenCalledWith('src')
})

it('reports autoplay rejection and releases the blob URL', async () => {
  play.mockRejectedValueOnce(new Error('NotAllowedError'))
  await expect(translationPlayer({ audio: async () => new Blob() })(result, new AbortController().signal, vi.fn()))
    .rejects.toThrow('재생하지 못했습니다')
  expect(revoke).toHaveBeenCalledWith('blob:test')
})

it('does not start a late audio download after stop', async () => {
  let resolve!: (blob: Blob) => void
  const audio = () => new Promise<Blob>((done) => { resolve = done })
  const controller = new AbortController()
  const task = translationPlayer({ audio })(result, controller.signal, vi.fn())
  controller.abort(); resolve(new Blob()); await task
  expect(createUrl).not.toHaveBeenCalled(); expect(play).not.toHaveBeenCalled()
})

it('uses the target language for browser fallback and cancels it on stop', async () => {
  const speak = vi.fn(), cancel = vi.fn()
  vi.stubGlobal('speechSynthesis', { speak, cancel })
  vi.stubGlobal('SpeechSynthesisUtterance', class {
    text: string; lang = ''; onend = null; onerror = null
    constructor(text: string) { this.text = text }
  })
  const controller = new AbortController(), playing = vi.fn()
  const task = translationPlayer({ audio: vi.fn() })({ ...result, audio_id: null }, controller.signal, playing)
  expect(speak.mock.calls[0][0]).toMatchObject({ text: 'Hello', lang: 'en-US' })
  expect(playing).toHaveBeenCalledOnce()
  controller.abort(); await task
  expect(cancel).toHaveBeenCalledOnce()
})

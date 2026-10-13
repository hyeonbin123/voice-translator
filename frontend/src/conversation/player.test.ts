// [파일 역할] player.ts(translationPlayer) 테스트 묶음. 서버 음성 재생과 브라우저 내장 음성 대체 재생에서
// 시작·끝·멈춤·오류 처리와 blob URL 해제를 확인한다.
// [가짜 객체] jsdom은 소리를 재생하지 못하므로 Audio 요소, URL.createObjectURL·revokeObjectURL,
// speechSynthesis를 가짜로 바꾼다. 만들어진 Audio 요소를 elements에 모아 테스트가 onended·onerror를 직접
// 일으킨다.

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

// 확인: audio_id로 인증된 음성을 받아(api.audio) 재생을 시작하고 playing을 한 번 부르며, 정상 종료·멈춤·재생
// 오류 세 경우 모두 재생을 멈추고 src를 지우고 blob URL을 해제한다. 오류일 때만 약속이 실패한다.
// 왜: 어느 경로로 끝나도 받은 음성이 메모리에 남지 않아야 하고, 멈춤은 오류로 보이면 안 된다.
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

// 확인: 브라우저가 자동 재생을 거절하면(play() 실패) 재생 실패 안내로 끝나고 blob URL은 해제된다.
it('reports autoplay rejection and releases the blob URL', async () => {
  play.mockRejectedValueOnce(new Error('NotAllowedError'))
  await expect(translationPlayer({ audio: async () => new Blob() })(result, new AbortController().signal, vi.fn()))
    .rejects.toThrow('재생하지 못했습니다')
  expect(revoke).toHaveBeenCalledWith('blob:test')
})

// 확인: 음성을 받는 동안 멈추면, 늦게 도착한 음성으로 URL을 만들거나 재생하지 않는다.
// 왜: 사용자가 멈춘 뒤 소리가 갑자기 나는 일을 막는다.
it('does not start a late audio download after stop', async () => {
  let resolve!: (blob: Blob) => void
  const audio = () => new Promise<Blob>((done) => { resolve = done })
  const controller = new AbortController()
  const task = translationPlayer({ audio })(result, controller.signal, vi.fn())
  controller.abort(); resolve(new Blob()); await task
  expect(createUrl).not.toHaveBeenCalled(); expect(play).not.toHaveBeenCalled()
})

// 확인: 서버 음성이 없을 때 번역문을 도착 언어(en-US)로 읽고, 멈추면 speechSynthesis.cancel로 읽기를 끊는다.
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

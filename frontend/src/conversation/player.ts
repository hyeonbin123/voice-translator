import type { TranslationApi, TranslationResult } from '../translation/api'

export type PlayTranslation = (result: TranslationResult, signal: AbortSignal, playing: () => void) => Promise<void>
export const PLAYBACK_ERROR = '번역 음성을 재생하지 못했습니다. 음성 다시 듣기를 눌러 주세요.'

export function translationPlayer(api: Pick<TranslationApi, 'audio'>): PlayTranslation {
  return async (result, signal, playing) => {
    if (result.audio_id) {
      const blob = await api.audio(result.audio_id, signal)
      if (signal.aborted) return
      const url = URL.createObjectURL(blob)
      const audio = new Audio(url)
      try {
        await new Promise<void>((resolve, reject) => {
          const finish = (error?: Error) => {
            signal.removeEventListener('abort', abort)
            audio.onended = audio.onerror = null
            audio.pause()
            if (error) reject(error); else resolve()
          }
          const abort = () => finish()
          signal.addEventListener('abort', abort, { once: true })
          audio.onended = () => finish()
          audio.onerror = () => finish(new Error(PLAYBACK_ERROR))
          playing()
          void audio.play().catch(() => finish(new Error(PLAYBACK_ERROR)))
        })
      } finally {
        audio.removeAttribute('src')
        URL.revokeObjectURL(url)
      }
    } else {
      if (signal.aborted) return
      if (!window.speechSynthesis || typeof SpeechSynthesisUtterance === 'undefined') throw new Error(PLAYBACK_ERROR)
      await new Promise<void>((resolve, reject) => {
        const utterance = new SpeechSynthesisUtterance(result.translated_text)
        utterance.lang = result.target_lang === 'ko' ? 'ko-KR' : 'en-US'
        const finish = (error?: Error) => {
          signal.removeEventListener('abort', abort)
          utterance.onend = utterance.onerror = null
          if (error) reject(error); else resolve()
        }
        const abort = () => { finish(); window.speechSynthesis.cancel() }
        signal.addEventListener('abort', abort, { once: true })
        utterance.onend = () => finish()
        utterance.onerror = () => finish(new Error(PLAYBACK_ERROR))
        playing()
        try { window.speechSynthesis.speak(utterance) } catch { finish(new Error(PLAYBACK_ERROR)) }
      })
    }
  }
}

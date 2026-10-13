// [파일 역할] 대화 모드·동시통역·두 사람 대화에서 번역 결과를 소리로 읽어 주는 재생 함수. 재생이 끝날 때까지
// 기다리는 약속(Promise)을 돌려줘, 세션이 재생 동안 마이크를 쉬게 하고 다음 마디를 순서대로 재생할 수 있게
// 한다.
// [흐름] 서버 음성이 있으면(audio_id) GET /api/audio/{id}로 받아 blob URL로 재생, 없으면(합성 실패·모델 없음)
// 브라우저 내장 음성(speechSynthesis)으로 읽는다(docs/api.md).
// [연결] 세 세션(conversation·live·dialog의 session.ts)의 기본 재생 함수다. 테스트는 PlayTranslation 타입의
// 가짜 함수를 넣는다.

import type { TranslationApi, TranslationResult } from '../translation/api'

// result: 읽을 번역 결과, signal: 멈춤(취소), playing: 실제로 소리가 나기 시작할 때 부를 함수(세션이 '재생
// 중' 상태로 바꾸는 데 쓴다).
export type PlayTranslation = (result: TranslationResult, signal: AbortSignal, playing: () => void) => Promise<void>
// 재생 실패 안내. 세션은 이 문구를 그 마디의 audioError로 표시한다.
export const PLAYBACK_ERROR = '번역 음성을 재생하지 못했습니다. 음성 다시 듣기를 눌러 주세요.'

// 음성 받기 API만 있으면 되는 재생 함수를 만든다.
export function translationPlayer(api: Pick<TranslationApi, 'audio'>): PlayTranslation {
  return async (result, signal, playing) => {
    if (result.audio_id) {
      // <audio src>로는 인증 머리글을 보낼 수 없어 fetch로 받은 Blob에 임시 blob URL을 만든다. 받는 동안
      // 멈췄으면 재생하지 않는다.
      const blob = await api.audio(result.audio_id, signal)
      if (signal.aborted) return
      const url = URL.createObjectURL(blob)
      const audio = new Audio(url)
      try {
        await new Promise<void>((resolve, reject) => {
          // 끝(정상 종료, 멈춤, 오류) 처리를 한곳에 모은다. 처리기를 떼고 재생을 멈춘 뒤 약속을 끝낸다.
          // 멈춤(abort)은 오류가 아니라 정상 종료로 본다.
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
          // 자동 재생 정책 등으로 play()가 거절되면 재생 실패로 처리한다.
          void audio.play().catch(() => finish(new Error(PLAYBACK_ERROR)))
        })
      } finally {
        // 다 쓴 blob URL을 해제해 받은 음성이 메모리에 남지 않게 한다(docs/api.md: 다 쓰면
        // URL.revokeObjectURL).
        audio.removeAttribute('src')
        URL.revokeObjectURL(url)
      }
    } else {
      if (signal.aborted) return
      // 서버 음성이 없을 때: 브라우저 내장 음성 합성으로 번역문을 읽는다. 기능이 없는 브라우저면 재생 실패다.
      if (!window.speechSynthesis || typeof SpeechSynthesisUtterance === 'undefined') throw new Error(PLAYBACK_ERROR)
      await new Promise<void>((resolve, reject) => {
        const utterance = new SpeechSynthesisUtterance(result.translated_text)
        // 도착 언어에 맞는 음성을 고르게 언어 태그를 지정한다.
        utterance.lang = result.target_lang === 'ko' ? 'ko-KR' : 'en-US'
        const finish = (error?: Error) => {
          signal.removeEventListener('abort', abort)
          utterance.onend = utterance.onerror = null
          if (error) reject(error); else resolve()
        }
        // 멈추면 읽던 내용도 취소한다. speechSynthesis는 페이지 전체가 함께 쓰는 대기열이라 cancel하지 않으면
        // 계속 읽는다.
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

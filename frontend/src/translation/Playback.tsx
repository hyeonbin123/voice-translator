// [파일 역할] 번역 결과 하나의 음성 재생 영역. 서버 음성이 있으면 인증된 요청으로 받아 <audio controls>로
// 재생하고, 없으면(합성 실패·합성 모델 없음) 브라우저 내장 음성으로 번역문을 읽는 버튼을 준다(README 설계
// 판단: 합성이 실패해도 번역은 돌려준다).
// [설계 이유] <audio src>는 인증 머리글을 보낼 수 없어 fetch로 받은 Blob을 blob URL로 바꿔 쓰고, 화면을
// 떠나거나 결과가 바뀌면 URL을 해제한다(docs/api.md GET /api/audio/{id}). 자동 재생은 하지 않고 사용자가
// 컨트롤로 재생한다(대화 모드의 자동 재생은 conversation/player.ts).
// [연결] TranslatePage.tsx의 번역 결과와 history/HistoryPage.tsx의 기록 상세가 쓴다. 둘 다 key로 결과 id를 줘
// 결과가 바뀌면 새로 그린다.

import { useEffect, useRef, useState } from 'react'
import { errorMessage } from '../api/client'
import { TranslationApi, type TranslationResult } from './api'

// 재생에 필요한 필드만 받는다. 기록 항목에는 tts_error가 없으므로(기록에 저장하지 않음) 선택 필드로 둔다.
type PlaybackResult = Pick<TranslationResult, 'audio_id' | 'translated_text' | 'target_lang'> & Partial<Pick<TranslationResult, 'tts_error'>>

// 상태: url(받은 음성의 blob URL), error, loading(다시 불러오는 중), retry(다시 불러오기 횟수, 바뀌면
// effect가 다시 돈다), speaking(브라우저 음성으로 읽는 중). utterance ref는 정리할 때 읽기를 멈추려고 쥐고
// 있는다.
export default function Playback({ result, api }: { result: PlaybackResult; api: Pick<TranslationApi, 'audio'> }) {
  const audio = useRef<HTMLAudioElement>(null)
  const [url, setUrl] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [retry, setRetry] = useState(0)
  const [speaking, setSpeaking] = useState(false)
  const utterance = useRef<SpeechSynthesisUtterance | null>(null)

  // 음성 받기: audio_id가 있으면 받아서 blob URL을 만든다. 취소된 요청의 결과는 버린다.
  // 정리 함수: 요청 취소, 재생 멈춤, blob URL 해제, 브라우저 음성 읽기 중지(처리기를 먼저 떼어 정리 뒤 상태가
  // 바뀌지 않게). 결과·api가 바뀌거나 다시 불러오기를 누르거나 화면을 떠날 때 돈다.
  useEffect(() => {
    const controller = new AbortController()
    let objectUrl = ''
    const player = audio.current
    if (result.audio_id) {
      void api.audio(result.audio_id, controller.signal).then((blob) => {
        if (controller.signal.aborted) return
        objectUrl = URL.createObjectURL(blob)
        setUrl(objectUrl)
        setLoading(false)
      }).catch((cause: unknown) => {
        if (controller.signal.aborted) return
        setError(errorMessage(cause))
        setLoading(false)
      })
    }
    return () => {
      controller.abort()
      player?.pause()
      if (objectUrl) URL.revokeObjectURL(objectUrl)
      if (utterance.current) {
        utterance.current.onend = null
        utterance.current.onerror = null
        window.speechSynthesis?.cancel()
        utterance.current = null
      }
    }
  }, [api, result.audio_id, retry])

  // 브라우저 내장 음성으로 번역문 읽기. 이전에 읽던 것을 먼저 취소하고, 도착 언어에 맞는 언어 태그를 단다.
  // 사용자가 중지했거나 다른 읽기로 끊긴 경우(canceled, interrupted)는 오류로 보이지 않는다.
  function speak() {
    setError('')
    if (!window.speechSynthesis || typeof SpeechSynthesisUtterance === 'undefined') {
      setError('이 브라우저에서는 음성 읽기를 사용할 수 없습니다. 번역문을 확인해 주세요.')
      return
    }
    window.speechSynthesis.cancel()
    const next = new SpeechSynthesisUtterance(result.translated_text)
    next.lang = result.target_lang === 'ko' ? 'ko-KR' : 'en-US'
    next.onend = () => setSpeaking(false)
    next.onerror = (event) => {
      setSpeaking(false)
      if (event.error !== 'canceled' && event.error !== 'interrupted') setError('음성을 읽지 못했습니다. 브라우저의 음성 설정을 확인해 주세요.')
    }
    utterance.current = next
    try {
      window.speechSynthesis.speak(next)
      setSpeaking(true)
    } catch {
      setSpeaking(false)
      setError('음성을 읽지 못했습니다. 다시 시도해 주세요.')
    }
  }

  return (
    <div className="playback">
      {/* 서버 음성이 있으면: 재생 컨트롤, 불러오는 중 상태, 받기나 재생에 실패했을 때 '음성 다시 */}
      {/* 불러오기'(retry를 올려 effect를 다시 돌림). 없으면: 합성 실패인지 음성이 없는지 알리고 '번역문 */}
      {/* 읽기'·'읽기 중지' 버튼. */}
      {result.audio_id ? <>
        <audio ref={audio} aria-label="번역 음성" controls src={url || undefined}
          onError={() => setError('번역 음성을 재생하지 못했습니다. 다시 불러와 주세요.')} />
        {!url && !error && <p role="status">번역 음성을 불러오는 중…</p>}
        {error && <button disabled={loading} onClick={() => {
          setError(''); setLoading(true); setUrl(''); setRetry((value) => value + 1)
        }}>음성 다시 불러오기</button>}
      </> : <>
        <p className="hint">{result.tts_error === 'Speech synthesis failed' ? '서버 음성 합성에 실패했습니다.' : '서버 음성이 없습니다.'} 브라우저 음성으로 읽을 수 있습니다.</p>
        <div className="actions">
          <button onClick={speak}>번역문 읽기</button>
          {speaking && <button onClick={() => { window.speechSynthesis.cancel(); setSpeaking(false) }}>읽기 중지</button>}
        </div>
      </>}
      {error && <p className="error" role="alert">{error}</p>}
    </div>
  )
}

import { useEffect, useRef, useState } from 'react'
import { errorMessage } from '../api/client'
import { TranslationApi, type TranslationResult } from './api'

export default function Playback({ result, api }: { result: TranslationResult; api: TranslationApi }) {
  const audio = useRef<HTMLAudioElement>(null)
  const [url, setUrl] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [retry, setRetry] = useState(0)
  const [speaking, setSpeaking] = useState(false)
  const utterance = useRef<SpeechSynthesisUtterance | null>(null)

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

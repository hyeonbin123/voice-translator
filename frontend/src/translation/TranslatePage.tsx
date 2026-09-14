import { useEffect, useRef, useState, type FormEvent } from 'react'
import { errorMessage } from '../api/client'
import { characterCount, MAX_AUDIO_BYTES, TranslationApi, type Language, type TranslationResult } from './api'
import Playback from './Playback'
import { useRecorder } from './useRecorder'
import ConversationPanel from '../conversation/ConversationPanel'
import LivePanel from '../live/LivePanel'

const languageName = { ko: '한국어', en: '영어' }
const timing = (value: number | null) => value === null ? '실행 안 함' : `${value.toLocaleString('ko-KR')} ms`

export default function TranslatePage({ api }: { api: TranslationApi }) {
  const [source, setSource] = useState<Language>('ko')
  const target = source === 'ko' ? 'en' : 'ko'
  const [text, setText] = useState('')
  const [mode, setMode] = useState<'text' | 'speech' | 'conversation' | 'live'>('text')
  const [conversationActive, setConversationActive] = useState(false)
  const [clip, setClip] = useState<Blob | null>(null)
  const [result, setResult] = useState<TranslationResult | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const pending = useRef<AbortController | null>(null)
  const resultHeading = useRef<HTMLHeadingElement>(null)
  const errorSummary = useRef<HTMLParagraphElement>(null)
  const recordingError = useRef<HTMLParagraphElement>(null)
  const recordingStart = useRef<HTMLButtonElement>(null)
  const recordingCancel = useRef<HTMLButtonElement>(null)
  const audioFile = useRef<HTMLInputElement>(null)
  const recorder = useRecorder(setClip)
  const recording = recorder.state !== 'idle'
  const count = characterCount(text)
  const previousRecording = useRef(recording)

  useEffect(() => () => { pending.current?.abort() }, [])
  useEffect(() => { if (result) resultHeading.current?.focus() }, [result])
  useEffect(() => { if (error) errorSummary.current?.focus() }, [error])
  useEffect(() => { if (recorder.error) recordingError.current?.focus() }, [recorder.error])
  useEffect(() => {
    if (!previousRecording.current && recording) recordingCancel.current?.focus()
    if (previousRecording.current && !recording && !recorder.error) recordingStart.current?.focus()
    previousRecording.current = recording
  }, [recording, recorder.error])

  async function submit(event: FormEvent) {
    event.preventDefault()
    if (pending.current || recording || mode === 'conversation' || mode === 'live') return
    setError('')
    if (mode === 'text' && (count < 1 || count > 500)) {
      setError('앞뒤 공백을 제외하고 1~500자를 입력해 주세요.')
      return
    }
    if (mode === 'speech' && (!clip || !clip.size || clip.size > MAX_AUDIO_BYTES)) {
      setError('녹음하거나 10MB 이하의 음성 파일을 선택해 주세요.')
      return
    }
    const controller = new AbortController()
    pending.current = controller
    setBusy(true)
    setResult(null)
    try {
      const direction = { source_lang: source, target_lang: target } as const
      const next = mode === 'text'
        ? await api.text(text, direction, controller.signal)
        : await api.speech(clip!, direction, controller.signal)
      if (!controller.signal.aborted) setResult(next)
    } catch (cause) {
      if (!controller.signal.aborted) setError(errorMessage(cause))
    } finally {
      if (!controller.signal.aborted) {
        pending.current = null
        setBusy(false)
      }
    }
  }

  return (
    <section className="translator">
      <p className="eyebrow">영어 ↔ 한국어</p>
      <h1 tabIndex={-1}>번역</h1>
      <p className="muted">말하거나 글을 입력해 대화를 이어 가세요.</p>
      {api.demo && <aside className="notice">
        <strong>예시 모드</strong> · 예시 번역을 보여 주며 기록은 저장하지 않습니다.
        <br />한국어는 “안녕하세요”, “감사합니다”, 영어는 “Hello”, “Thank you”를 입력해 보세요.
        녹음 결과도 고정된 인사말 예시입니다.
      </aside>}
      <form onSubmit={(event) => { void submit(event) }} aria-busy={busy}>
        <fieldset disabled={busy || recording || conversationActive} className="direction">
          <legend>번역 방향</legend>
          <label htmlFor="source-language">말하거나 입력할 언어</label>
          <select id="source-language" value={source} onChange={(event) => {
            setSource(event.target.value as Language); setClip(null); setError('')
          }}><option value="ko">한국어 → 영어</option><option value="en">영어 → 한국어</option></select>
        </fieldset>
        <fieldset disabled={busy || recording || conversationActive} className="mode-picker">
          <legend>입력 방법</legend>
          <label><input type="radio" name="input-mode" checked={mode === 'text'} onChange={() => { setMode('text'); setError('') }} /> 글자 입력</label>
          <label><input type="radio" name="input-mode" checked={mode === 'speech'} onChange={() => { setMode('speech'); setError('') }} /> 음성 입력</label>
          <label><input type="radio" name="input-mode" checked={mode === 'conversation'} onChange={() => { setMode('conversation'); setError(''); setResult(null) }} /> 대화 모드</label>
          <label><input type="radio" name="input-mode" checked={mode === 'live'} onChange={() => { setMode('live'); setError(''); setResult(null) }} /> 동시통역</label>
        </fieldset>
        {mode === 'live' ? <LivePanel api={api} direction={{ source_lang: source, target_lang: target }} onActiveChange={setConversationActive} /> : mode === 'conversation' ? <ConversationPanel api={api} direction={{ source_lang: source, target_lang: target }} onActiveChange={setConversationActive} /> : mode === 'text' ? <>
          <label htmlFor="source-text">번역할 글 ({languageName[source]})</label>
          <textarea id="source-text" value={text} disabled={busy} rows={6} aria-describedby={`text-count${error ? ' translation-error' : ''}`}
            aria-invalid={count > 500 || (!!error && count === 0)} onChange={(event) => setText(event.target.value)} placeholder="번역할 내용을 입력하세요" />
          <p id="text-count" className={count > 500 ? 'error' : 'hint'}>{count} / 500자 · 앞뒤 공백 제외</p>
        </> : <div className="recording-panel">
          <p id="audio-hint" className="hint">{languageName[source]}로 말해 주세요. 최대 30초 · 10MB. 녹음 원본은 저장하지 않습니다.</p>
          <div className="actions">
            {recorder.state === 'idle' ? <button ref={recordingStart} type="button" disabled={busy} onClick={() => { setClip(null); setError(''); void recorder.start() }}>녹음 시작</button> : <>
              {recorder.state === 'recording' && <button type="button" onClick={recorder.stop}>녹음 끝내기</button>}
              <button ref={recordingCancel} type="button" onClick={recorder.cancel}>녹음 취소</button>
            </>}
          </div>
          {recorder.state === 'permission' && <p role="status">마이크 권한을 기다리는 중…</p>}
          {recorder.state === 'recording' && <p role="status">녹음 중 · {recorder.seconds} / 30초</p>}
          {recorder.notice && <p role="status">{recorder.notice}</p>}
          {recorder.error && <p ref={recordingError} tabIndex={-1} role="alert" className="error">{recorder.error}</p>}
          <button type="button" disabled={busy || recording} onClick={() => audioFile.current?.click()}
            aria-describedby={`audio-hint audio-selection${error ? ' translation-error' : ''}`}>
            또는 음성 파일 선택
          </button>
          <input ref={audioFile} id="audio-file" type="file" hidden aria-label="음성 파일"
            accept="audio/*,.webm,.wav,.ogg,.mp3,.m4a" disabled={busy || recording}
            onChange={(event) => {
              const file = event.target.files?.[0]
              if (!file) return
              setClip(null); setError('')
              if (file.size === 0 || file.size > MAX_AUDIO_BYTES) setError('비어 있지 않은 10MB 이하 음성 파일을 선택해 주세요.')
              else setClip(file)
              event.target.value = ''
            }} />
          <p id="audio-selection" role="status" aria-atomic="true">
            {clip ? <>{clip instanceof File ? `선택한 파일: ${clip.name}` : '녹음 완료'} · {(clip.size / 1000).toFixed(1)} KB · 번역할 준비가 되었습니다.</> : '선택한 음성이 없습니다.'}
          </p>
        </div>}
        {mode !== 'conversation' && mode !== 'live' && <button className="primary" type="submit" disabled={busy || recording || (mode === 'speech' && !clip)}>
          {busy ? '번역 중…' : '번역하기'}
        </button>}
        {busy && <p role="status">번역과 음성을 준비하고 있습니다.</p>}
        {error && <p id="translation-error" ref={errorSummary} tabIndex={-1} className="error" role="alert">{error}</p>}
      </form>
      {mode !== 'conversation' && mode !== 'live' && !result && !busy && <p className="hint">번역하면 이곳에 원문과 번역문이 표시됩니다. 번역 음성도 재생할 수 있습니다.</p>}
      {result && <section className="translation-result" aria-labelledby="result-title">
        <h2 id="result-title" ref={resultHeading} tabIndex={-1}>번역 결과</h2>
        <div className="result-columns">
          <div><h3>원문 · {languageName[result.source_lang]}</h3><p lang={result.source_lang}>{result.source_text}</p></div>
          <div><h3>번역문 · {languageName[result.target_lang]}</h3><p lang={result.target_lang}>{result.translated_text}</p></div>
        </div>
        <Playback key={result.id} result={result} api={api} />
        <dl className="timings">
          <div><dt>음성 인식</dt><dd>{timing(result.stt_ms)}</dd></div>
          <div><dt>번역</dt><dd>{timing(result.mt_ms)}</dd></div>
          <div><dt>음성 합성</dt><dd>{timing(result.tts_ms)}</dd></div>
        </dl>
        <p className="hint">각 단계의 처리 시간입니다. 서버 대기와 전송 시간은 포함하지 않습니다.</p>
      </section>}
    </section>
  )
}

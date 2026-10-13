// [파일 역할] 번역 화면(/translate). 번역 방향과 입력 방법 다섯 가지를 고르고, 글자 입력·음성 입력(녹음 또는
// 파일)은 이 화면이 직접 요청해 결과(원문, 번역문, 번역 음성, 단계별 처리 시간)를 보여 준다. 대화 모드·두
// 사람 대화·동시통역은 각 패널 컴포넌트에 맡긴다.
// [설계 이유] 키보드·스크린리더 사용자를 위해 결과가 나오면 결과 제목으로, 오류가 나면 오류 문구로, 녹음을
// 시작·끝내면 남아 있는 녹음 버튼으로 포커스를 옮긴다(faf1e55). 누른 버튼이 사라져 포커스가 길을 잃지 않게
// 하려는 것이다.
// [연결] App.tsx가 TranslationApi(실제 또는 예시 모드)를 넘겨 그린다. 녹음은 useRecorder.ts, 번역 음성 재생은
// Playback.tsx, 서버 호출은 api.ts를 쓴다.

import { useEffect, useRef, useState, type FormEvent } from 'react'
import { errorMessage } from '../api/client'
import { characterCount, MAX_AUDIO_BYTES, TranslationApi, type Language, type TranslationResult } from './api'
import Playback from './Playback'
import { useRecorder } from './useRecorder'
import ConversationPanel from '../conversation/ConversationPanel'
import LivePanel from '../live/LivePanel'
import DialogPanel from '../dialog/DialogPanel'

// languageName: 언어 코드의 화면 이름. timing: 단계별 처리 시간(ms)을 보여 주되, 실행하지 않은 단계(글자
// 입력의 음성 인식, 합성 실패)는 null이라 '실행 안 함'으로 쓴다.
const languageName = { ko: '한국어', en: '영어' }
const timing = (value: number | null) => value === null ? '실행 안 함' : `${value.toLocaleString('ko-KR')} ms`

// 상태: source(말하거나 입력할 언어, 도착 언어 target은 그 반대로 정해짐), text, mode(입력 방법),
// conversationActive(세 패널 중 하나가 켜져 있는지), clip(녹음이나 고른 파일), result, error, busy.
// ref: pending은 진행 중인 요청의 AbortController(화면을 떠나면 취소), 나머지는 포커스를 옮길 대상
// 요소들이다.
export default function TranslatePage({ api }: { api: TranslationApi }) {
  const [source, setSource] = useState<Language>('ko')
  const target = source === 'ko' ? 'en' : 'ko'
  const [text, setText] = useState('')
  const [mode, setMode] = useState<'text' | 'speech' | 'conversation' | 'dialog' | 'live'>('text')
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

  // 화면을 떠나면 진행 중인 번역 요청을 취소한다. 그 아래 세 effect는 결과·오류·녹음 오류가 생길 때 해당
  // 요소로 포커스를 옮긴다.
  useEffect(() => () => { pending.current?.abort() }, [])
  useEffect(() => { if (result) resultHeading.current?.focus() }, [result])
  useEffect(() => { if (error) errorSummary.current?.focus() }, [error])
  useEffect(() => { if (recorder.error) recordingError.current?.focus() }, [recorder.error])
  // 녹음 버튼 포커스 관리: 녹음이 시작되면 '녹음 시작' 버튼이 사라지므로 '녹음 취소' 버튼으로, 녹음이
  // 끝나면(오류 없이) 다시 '녹음 시작' 버튼으로 포커스를 옮긴다. 직전 녹음 여부를 ref로 기억해 바뀐 순간만
  // 처리한다.
  useEffect(() => {
    if (!previousRecording.current && recording) recordingCancel.current?.focus()
    if (previousRecording.current && !recording && !recorder.error) recordingStart.current?.focus()
    previousRecording.current = recording
  }, [recording, recorder.error])

  // 글자·음성 번역 제출. 이미 요청 중이거나 녹음 중이거나 패널 방식이면 무시한다. 서버 규칙과 같은 검사를
  // 먼저 해 헛 요청을 줄인다: 글은 앞뒤 공백 제외 1~500자, 음성은 비어 있지 않고 크기 한도 이하. 요청마다 새
  // AbortController를 두고, 취소된 요청의 결과·오류는 화면에 반영하지 않는다.
  async function submit(event: FormEvent) {
    event.preventDefault()
    if (pending.current || recording || mode === 'conversation' || mode === 'dialog' || mode === 'live') return
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
      {/* 예시 모드 안내: 서버 없이 정해진 인사말만 번역되고 기록이 남지 않는다는 것을 분명히 보인다. */}
      {api.demo && <aside className="notice">
        <strong>예시 모드</strong> · 예시 번역을 보여 주며 기록은 저장하지 않습니다.
        <br />한국어는 “안녕하세요”, “감사합니다”, 영어는 “Hello”, “Thank you”를 입력해 보세요.
        녹음 결과도 고정된 인사말 예시입니다.
      </aside>}
      <form onSubmit={(event) => { void submit(event) }} aria-busy={busy}>
        {/* 번역 방향 고르기. 두 사람 대화는 서버가 마디마다 언어를 가리므로 숨긴다. 요청·녹음 중이거나 */}
        {/* 패널이 켜져 있으면 바꿀 수 없게 잠근다(진행 중인 대화와 화면의 방향이 어긋나지 않게). */}
        {mode !== 'dialog' && <fieldset disabled={busy || recording || conversationActive} className="direction">
          <legend>번역 방향</legend>
          <label htmlFor="source-language">말하거나 입력할 언어</label>
          <select id="source-language" value={source} onChange={(event) => {
            setSource(event.target.value as Language); setClip(null); setError('')
          }}><option value="ko">한국어 → 영어</option><option value="en">영어 → 한국어</option></select>
        </fieldset>}
        {/* 입력 방법 다섯 가지(라디오 버튼). 패널 방식으로 바꾸면 이전 결과를 지운다. 패널이 켜져 있는 */}
        {/* 동안은 바꿀 수 없다. */}
        <fieldset disabled={busy || recording || conversationActive} className="mode-picker">
          <legend>입력 방법</legend>
          <label><input type="radio" name="input-mode" checked={mode === 'text'} onChange={() => { setMode('text'); setError('') }} /> 글자 입력</label>
          <label><input type="radio" name="input-mode" checked={mode === 'speech'} onChange={() => { setMode('speech'); setError('') }} /> 음성 입력</label>
          <label><input type="radio" name="input-mode" checked={mode === 'dialog'} onChange={() => { setMode('dialog'); setError(''); setResult(null) }} /> 두 사람 대화</label>
          <label><input type="radio" name="input-mode" checked={mode === 'conversation'} onChange={() => { setMode('conversation'); setError(''); setResult(null) }} /> 대화 모드</label>
          <label><input type="radio" name="input-mode" checked={mode === 'live'} onChange={() => { setMode('live'); setError(''); setResult(null) }} /> 동시통역</label>
        </fieldset>
        {/* 입력 방법별 영역: 동시통역·두 사람 대화·대화 모드는 각 패널, 글자 입력은 글 상자와 글자 수, */}
        {/* 음성 입력은 녹음·파일 선택 영역. 패널에는 켜짐 여부를 받는 setConversationActive를 넘긴다. */}
        {mode === 'live' ? <LivePanel api={api} direction={{ source_lang: source, target_lang: target }} onActiveChange={setConversationActive} /> : mode === 'dialog' ? <DialogPanel api={api} onActiveChange={setConversationActive} /> : mode === 'conversation' ? <ConversationPanel api={api} direction={{ source_lang: source, target_lang: target }} onActiveChange={setConversationActive} /> : mode === 'text' ? <>
          <label htmlFor="source-text">번역할 글 ({languageName[source]})</label>
          {/* 글자 수 안내와 오류를 aria-describedby로 묶어 함께 읽히게 하고, 500자를 넘거나 빈 글로 */}
          {/* 오류가 났을 때 aria-invalid로 표시한다. */}
          <textarea id="source-text" value={text} disabled={busy} rows={6} aria-describedby={`text-count${error ? ' translation-error' : ''}`}
            aria-invalid={count > 500 || (!!error && count === 0)} onChange={(event) => setText(event.target.value)} placeholder="번역할 내용을 입력하세요" />
          <p id="text-count" className={count > 500 ? 'error' : 'hint'}>{count} / 500자 · 앞뒤 공백 제외</p>
        </> : <div className="recording-panel">
          <p id="audio-hint" className="hint">{languageName[source]}로 말해 주세요. 최대 30초 · 10MB. 녹음 원본은 저장하지 않습니다.</p>
          {/* 녹음 버튼: 대기 중이면 '녹음 시작', 녹음 중이면 '녹음 끝내기'와 '녹음 취소'(권한을 기다리는 */}
          {/* 중에는 취소만). 시작하면 이전 녹음과 오류를 지운다. */}
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
          {/* 파일 선택: 기본 파일 입력은 숨기고 같은 모양의 버튼으로 연다. 고른 파일이 비었거나 크기 */}
          {/* 한도를 넘으면 오류를 보이고, 고른 뒤 입력값을 비워 같은 파일을 다시 골라도 onChange가 불리게 */}
          {/* 한다. accept는 MediaRecorder 기본 형식(webm)과 서버가 풀 수 있는 형식을 고르기 쉽게 하는 */}
          {/* 힌트이며, 실제 확인은 서버가 디코딩으로 한다(docs/api.md). */}
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
          {/* 지금 번역할 음성이 무엇인지(파일 이름 또는 녹음 완료, 크기) 알려 주는 상태 문구. */}
          <p id="audio-selection" role="status" aria-atomic="true">
            {clip ? <>{clip instanceof File ? `선택한 파일: ${clip.name}` : '녹음 완료'} · {(clip.size / 1000).toFixed(1)} KB · 번역할 준비가 되었습니다.</> : '선택한 음성이 없습니다.'}
          </p>
        </div>}
        {/* '번역하기' 버튼은 글자·음성 입력에만 있다. 패널 방식은 각자 시작 버튼을 쓴다. */}
        {mode !== 'conversation' && mode !== 'dialog' && mode !== 'live' && <button className="primary" type="submit" disabled={busy || recording || (mode === 'speech' && !clip)}>
          {busy ? '번역 중…' : '번역하기'}
        </button>}
        {busy && <p role="status">번역과 음성을 준비하고 있습니다.</p>}
        {error && <p id="translation-error" ref={errorSummary} tabIndex={-1} className="error" role="alert">{error}</p>}
      </form>
      {mode !== 'conversation' && mode !== 'dialog' && mode !== 'live' && !result && !busy && <p className="hint">번역하면 이곳에 원문과 번역문이 표시됩니다. 번역 음성도 재생할 수 있습니다.</p>}
      {/* 번역 결과: 원문·번역문(lang으로 언어별 발음), 번역 음성 재생, 단계별 처리 시간. Playback에 */}
      {/* key={result.id}를 줘 새 결과마다 재생 상태를 새로 시작한다. 처리 시간은 모델 호출 시간만이고 */}
      {/* 서버 대기·전송 시간은 넣지 않는다(docs/api.md 저장과 삭제 절). */}
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

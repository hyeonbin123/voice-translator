import { useEffect, useMemo, useSyncExternalStore } from 'react'
import type { Direction, TranslationApi } from '../translation/api'
import { LiveSession, type Caption } from './session'

const turnStatus = { speaking: '자막 갱신 중', waiting: '최종 번역 중', done: '번역 완료', error: '번역 실패', canceled: '취소됨' }

export function Subtitle({ caption, lang }: { caption: Caption; lang: string }) {
  const characters = Array.from(caption.text)
  return <p lang={lang} className="live-caption">
    {caption.stable > 0 && <><span className="sr-only">확정된 부분: </span><strong>{characters.slice(0, caption.stable).join('')}</strong></>}
    {caption.stable < characters.length && <><span className="sr-only">갱신 중인 부분: </span><span className="live-draft">{characters.slice(caption.stable).join('')}</span></>}
    {!characters.length && <span className="hint">자막을 기다리는 중…</span>}
  </p>
}

export default function LivePanel({ api, direction, onActiveChange }: {
  api: TranslationApi; direction: Direction; onActiveChange: (active: boolean) => void
}) {
  const session = useMemo(() => new LiveSession(api), [api])
  const state = useSyncExternalStore(session.subscribe, session.getSnapshot)
  useEffect(() => () => { session.stop() }, [session])
  useEffect(() => { onActiveChange(state.active) }, [state.active, onActiveChange])
  const status = state.permission ? '마이크와 동시통역 연결 준비 중 · 처음에는 잠시 걸릴 수 있습니다'
    : state.playing ? '재생 중 · 듣기 멈춤'
    : state.active ? !state.ready ? '다시 연결 중 · 듣기 멈춤' : state.speaking ? '말소리 감지 · 자막 갱신 중' : '듣는 중' : '멈춤'

  return <section className="conversation-panel" aria-labelledby="live-title">
    <h2 id="live-title">동시통역</h2>
    <p id="live-hint" className="hint">{direction.source_lang === 'ko' ? '한국어' : '영어'}로 말하면 원문과 번역문이 자막으로 나타납니다. 약 1초 쉬면 최종 번역과 음성이 나옵니다. 음성 재생 중에는 듣기를 멈춥니다.</p>
    <p className="hint">진한 글자는 확정된 부분, 흐린 글자는 갱신 중인 부분입니다. 원문이 늘면 번역이 바뀔 수 있어 번역문은 최종 결과까지 흐리게 표시합니다.</p>
    <div className="actions">
      <button type="button" className="primary" aria-describedby="live-hint" aria-pressed={state.active} disabled={api.demo}
        onClick={() => { if (state.active) session.stop(); else void session.start(direction) }}>
        {state.active ? '동시통역 멈춤' : '동시통역 시작'}
      </button>
      {!state.active && state.playingId !== null && <button type="button" onClick={session.stop}>음성 멈춤</button>}
    </div>
    {api.demo && <p className="hint">동시통역은 로그인 후 사용할 수 있습니다.</p>}
    <p role="status" aria-atomic="true">{status}</p>
    {state.notice && <p role="status" className="hint">{state.notice}</p>}
    {state.error && <p role="alert" className="error">{state.error}</p>}
    <p className="hint">멈추면 마이크와 연결을 닫고 대기 중인 말과 음성을 취소합니다. 완료된 번역은 아래에 남습니다.</p>
    <div className="sr-only" role="status" aria-live="polite" aria-atomic="true">
      {state.announcement}
    </div>
    <ol className="conversation-turns" aria-label="동시통역 내용" aria-live="off">
      {state.turns.map((turn) => <li key={turn.id}>
        <h3>{turn.id}번째 말 · {turnStatus[turn.state]}</h3>
        <div className="result-columns">
          <div><h4>원문</h4><Subtitle caption={turn.source} lang={turn.direction.source_lang} /></div>
          <div><h4>번역문</h4><Subtitle caption={turn.translation} lang={turn.direction.target_lang} /></div>
        </div>
        {turn.result && <>
          {!turn.result.audio_id && <p className="hint">서버 음성이 없어 브라우저 음성으로 읽습니다.</p>}
          <button type="button" disabled={state.playingId === turn.id} onClick={() => session.replay(turn.id)}
            aria-label={`${turn.id}번째 번역 음성 다시 듣기`}>{state.playingId === turn.id ? '음성 준비·재생 중' : '음성 다시 듣기'}</button>
        </>}
        {turn.error && <p role="alert" className="error">{turn.error} 다음 말은 계속 번역합니다.</p>}
        {turn.audioError && <p role="alert" className="error">{turn.audioError}</p>}
      </li>)}
    </ol>
    {!state.turns.length && <p className="hint">듣는 중 표시가 나오면 말해 주세요.</p>}
  </section>
}

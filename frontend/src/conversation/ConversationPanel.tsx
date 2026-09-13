import { useEffect, useMemo, useSyncExternalStore } from 'react'
import type { Direction, TranslationApi } from '../translation/api'
import { ConversationSession } from './session'

const turnStatus = { queued: '번역 대기', translating: '번역 중', done: '번역 완료', error: '번역 실패', canceled: '멈춤 · 전송 취소' }

export default function ConversationPanel({ api, direction, onActiveChange }: {
  api: TranslationApi; direction: Direction; onActiveChange: (active: boolean) => void
}) {
  const session = useMemo(() => new ConversationSession(api), [api])
  const state = useSyncExternalStore(session.subscribe, session.getSnapshot)
  useEffect(() => () => { session.stop() }, [session])
  useEffect(() => { onActiveChange(state.active) }, [state.active, onActiveChange])
  const status = state.permission ? '마이크와 말소리 감지 준비 중 · 처음에는 잠시 걸릴 수 있습니다'
    : state.playing ? '재생 중 · 듣기 멈춤'
    : state.active ? `${state.speaking ? '말소리 감지' : '듣는 중'}${state.translating ? ' · 번역 중' : ''}` : '멈춤'

  return <section className="conversation-panel" aria-labelledby="conversation-title">
    <h2 id="conversation-title">대화 모드</h2>
    <p id="conversation-hint" className="hint">{direction.source_lang === 'ko' ? '한국어' : '영어'}로 말하고 잠시 쉬면 자동으로 번역합니다. 번역 중에도 듣고, 음성 재생 중에는 듣기를 멈춥니다.</p>
    <p className="hint">약 1초 동안 말소리가 없으면 한 마디를 보냅니다. 듣는 중 표시가 나오면 말해 주세요.</p>
    <div className="actions">
      <button type="button" className="primary" aria-describedby="conversation-hint" aria-pressed={state.active}
        onClick={() => { if (state.active) session.stop(); else void session.start(direction) }}>
        {state.active ? '대화 멈춤' : '대화 시작'}
      </button>
      {!state.active && state.playingId !== null && <button type="button" onClick={session.stop}>음성 멈춤</button>}
    </div>
    <p role="status" aria-atomic="true">{status}</p>
    {state.notice && <p role="status" className="hint">{state.notice}</p>}
    {state.error && <p role="alert" className="error">{state.error}</p>}
    <p className="hint">멈추면 아직 끝나지 않은 말과 대기 중인 번역·재생을 취소합니다. 완료된 번역은 아래에 남습니다.</p>
    <ol className="conversation-turns" aria-label="대화 내용" aria-live="polite" aria-relevant="additions text">
      {state.turns.map((turn) => <li key={turn.id}>
        <h3>{turn.id}번째 말 · {turnStatus[turn.state]}</h3>
        {turn.result && <>
          <div className="result-columns">
            <div><h4>원문</h4><p lang={turn.result.source_lang}>{turn.result.source_text}</p></div>
            <div><h4>번역문</h4><p lang={turn.result.target_lang}>{turn.result.translated_text}</p></div>
          </div>
          {!turn.result.audio_id && <p className="hint">서버 음성이 없어 브라우저 음성으로 읽습니다.</p>}
          <button type="button" disabled={state.playingId === turn.id} onClick={() => session.replay(turn.id)}
            aria-label={`${turn.id}번째 번역 음성 다시 듣기`}>{state.playingId === turn.id ? '음성 준비·재생 중' : '음성 다시 듣기'}</button>
        </>}
        {turn.error && <p className="error">{turn.error} 다음 말은 계속 번역합니다.</p>}
        {turn.audioError && <p className="error">{turn.audioError}</p>}
      </li>)}
    </ol>
    {!state.turns.length && <p className="hint">대화를 시작하면 원문, 번역문과 번역 음성이 순서대로 쌓입니다.</p>}
  </section>
}

// [파일 역할] 번역 화면의 '대화 모드' 입력 방식 화면. 시작·멈춤 버튼, 지금 상태(준비 중·듣는 중·말소리
// 감지·번역 중·재생 중), 안내·오류, 그리고 말한 순서대로 쌓이는 마디 목록(원문, 번역문, 음성 다시 듣기)을
// 그린다.
// [설계 이유] 마이크·판정·요청·재생의 복잡한 흐름은 ConversationSession(session.ts)이 React 밖에서 관리하고,
// 이 컴포넌트는 그 상태를 구독해 그리기만 한다. 그래서 흐름은 화면 없이 단위 테스트할 수 있다.
// [연결] translation/TranslatePage.tsx가 입력 방식으로 대화 모드를 고르면 이 컴포넌트를 그린다.

import { useEffect, useMemo, useSyncExternalStore } from 'react'
import type { Direction, TranslationApi } from '../translation/api'
import { ConversationSession } from './session'

// 마디 상태별 화면 문구.
const turnStatus = { queued: '번역 대기', translating: '번역 중', done: '번역 완료', error: '번역 실패', canceled: '멈춤 · 전송 취소' }

// api: 번역 API(실제 또는 예시), direction: 화면에서 고른 번역 방향, onActiveChange: 대화가 켜지고 꺼질 때
// 부모에게 알리는 함수.
export default function ConversationPanel({ api, direction, onActiveChange }: {
  api: TranslationApi; direction: Direction; onActiveChange: (active: boolean) => void
}) {
  // 세션은 api가 바뀔 때만 새로 만든다. useSyncExternalStore로 세션 상태가 바뀔 때마다 다시 그린다. 화면을
  // 떠나면(언마운트) 정리 함수가 stop을 불러 마이크·연결·재생을 모두 닫는다. 대화 중 여부는 부모에게 알린다.
  const session = useMemo(() => new ConversationSession(api), [api])
  const state = useSyncExternalStore(session.subscribe, session.getSnapshot)
  useEffect(() => () => { session.stop() }, [session])
  useEffect(() => { onActiveChange(state.active) }, [state.active, onActiveChange])
  // 한 줄 상태 문구. 준비 중(처음에는 판정 파일을 받느라 시간이 걸림) > 재생 중 > 대화 중(말소리 감지/듣는
  // 중, 번역 중 덧붙임) > 멈춤 순으로 고른다.
  const status = state.permission ? '마이크와 말소리 감지 준비 중 · 처음에는 잠시 걸릴 수 있습니다'
    : state.playing ? '재생 중 · 듣기 멈춤'
    : state.active ? `${state.speaking ? '말소리 감지' : '듣는 중'}${state.translating ? ' · 번역 중' : ''}` : '멈춤'

  return <section className="conversation-panel" aria-labelledby="conversation-title">
    <h2 id="conversation-title">대화 모드</h2>
    <p id="conversation-hint" className="hint">{direction.source_lang === 'ko' ? '한국어' : '영어'}로 말하고 잠시 쉬면 자동으로 번역합니다. 번역 중에도 듣고, 음성 재생 중에는 듣기를 멈춥니다.</p>
    <p className="hint">약 1초 동안 말소리가 없으면 한 마디를 보냅니다. 듣는 중 표시가 나오면 말해 주세요.</p>
    {/* 시작·멈춤 토글 버튼(aria-pressed로 켜짐 상태를 스크린리더에 알림). 대화를 멈춘 뒤 '음성 다시 */}
    {/* 듣기'로 재생 중일 때만 '음성 멈춤' 버튼이 나온다. */}
    <div className="actions">
      <button type="button" className="primary" aria-describedby="conversation-hint" aria-pressed={state.active}
        onClick={() => { if (state.active) session.stop(); else void session.start(direction) }}>
        {state.active ? '대화 멈춤' : '대화 시작'}
      </button>
      {!state.active && state.playingId !== null && <button type="button" onClick={session.stop}>음성 멈춤</button>}
    </div>
    {/* 상태 문구는 role="status"와 aria-atomic으로 바뀔 때마다 문장 전체를 다시 읽힌다. 짧은 안내는 */}
    {/* status, 대화를 멈추게 한 오류는 alert로 알린다. */}
    <p role="status" aria-atomic="true">{status}</p>
    {state.notice && <p role="status" className="hint">{state.notice}</p>}
    {state.error && <p role="alert" className="error">{state.error}</p>}
    <p className="hint">멈추면 아직 끝나지 않은 말과 대기 중인 번역·재생을 취소합니다. 완료된 번역은 아래에 남습니다.</p>
    {/* 마디 목록. aria-live="polite"와 aria-relevant="additions text"로 새 마디가 붙거나 결과 글이 */}
    {/* 채워지면 스크린리더가 말을 끊지 않고 읽어 준다. 원문·번역문 문단에 lang을 달아 각 언어의 발음으로 */}
    {/* 읽히게 한다. */}
    <ol className="conversation-turns" aria-label="대화 내용" aria-live="polite" aria-relevant="additions text">
      {state.turns.map((turn) => <li key={turn.id}>
        <h3>{turn.id}번째 말 · {turnStatus[turn.state]}</h3>
        {turn.result && <>
          <div className="result-columns">
            <div><h4>원문</h4><p lang={turn.result.source_lang}>{turn.result.source_text}</p></div>
            <div><h4>번역문</h4><p lang={turn.result.target_lang}>{turn.result.translated_text}</p></div>
          </div>
          {/* 서버 음성이 없으면(합성 실패·모델 없음) 브라우저 내장 음성으로 읽는다고 알린다(player.ts). */}
          {!turn.result.audio_id && <p className="hint">서버 음성이 없어 브라우저 음성으로 읽습니다.</p>}
          <button type="button" disabled={state.playingId === turn.id} onClick={() => session.replay(turn.id)}
            aria-label={`${turn.id}번째 번역 음성 다시 듣기`}>{state.playingId === turn.id ? '음성 준비·재생 중' : '음성 다시 듣기'}</button>
        </>}
        {/* 한 마디의 실패는 그 마디에만 표시하고 대화는 이어진다. */}
        {turn.error && <p className="error">{turn.error} 다음 말은 계속 번역합니다.</p>}
        {turn.audioError && <p className="error">{turn.audioError}</p>}
      </li>)}
    </ol>
    {!state.turns.length && <p className="hint">대화를 시작하면 원문, 번역문과 번역 음성이 순서대로 쌓입니다.</p>}
  </section>
}

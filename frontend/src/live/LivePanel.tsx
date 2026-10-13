// [파일 역할] 번역 화면의 '동시통역' 입력 방식 화면. 시작·멈춤 버튼, 연결·듣기 상태, 마디마다 원문·번역문
// 자막(진한 확정 부분 + 흐린 갱신 부분), 최종 결과의 음성 다시 듣기를 그린다.
// [설계 이유] 흐름은 LiveSession(session.ts)이 React 밖에서 관리하고 이 컴포넌트는 구독해 그리기만 한다.
// 자막은 1초마다 바뀌므로 목록은 스크린리더가 매번 읽지 않게 끄고(aria-live="off"), 최종 결과만 따로 한 번
// 읽힌다.
// [연결] translation/TranslatePage.tsx가 입력 방식으로 동시통역을 고르면 그린다. 예시 모드에서는 서버
// 웹소켓이 없어 버튼을 끈다.

import { useEffect, useMemo, useSyncExternalStore } from 'react'
import type { Direction, TranslationApi } from '../translation/api'
import { LiveSession, type Caption } from './session'

// 마디 상태별 화면 문구.
const turnStatus = { speaking: '자막 갱신 중', waiting: '최종 번역 중', done: '번역 완료', error: '번역 실패', canceled: '취소됨' }

// 자막 한 줄. 앞 stable 글자는 <strong>(진하게), 나머지는 흐린 글자로 그린다. 글자 단위는 서버와 같은
// 유니코드 코드 포인트라 Array.from으로 나눈다(한글·이모지가 반쪽으로 잘리지 않게). 색과 굵기만으로는
// 스크린리더가 차이를 알 수 없어 '확정된 부분'·'갱신 중인 부분'을 화면에 보이지 않는 글(sr-only)로 덧붙인다.
// lang으로 언어별 발음을 정한다.
export function Subtitle({ caption, lang }: { caption: Caption; lang: string }) {
  const characters = Array.from(caption.text)
  return <p lang={lang} className="live-caption">
    {caption.stable > 0 && <><span className="sr-only">확정된 부분: </span><strong>{characters.slice(0, caption.stable).join('')}</strong></>}
    {caption.stable < characters.length && <><span className="sr-only">갱신 중인 부분: </span><span className="live-draft">{characters.slice(caption.stable).join('')}</span></>}
    {!characters.length && <span className="hint">자막을 기다리는 중…</span>}
  </p>
}

// api: 번역 API, direction: 화면에서 고른 방향, onActiveChange: 켜짐 여부를 부모에게 알림. 세션은 api가 바뀔
// 때만 새로 만들고, 화면을 떠나면 stop으로 마이크·연결을 닫는다.
export default function LivePanel({ api, direction, onActiveChange }: {
  api: TranslationApi; direction: Direction; onActiveChange: (active: boolean) => void
}) {
  const session = useMemo(() => new LiveSession(api), [api])
  const state = useSyncExternalStore(session.subscribe, session.getSnapshot)
  useEffect(() => () => { session.stop() }, [session])
  useEffect(() => { onActiveChange(state.active) }, [state.active, onActiveChange])
  // 한 줄 상태: 준비 중 > 재생 중 > (켜짐) 다시 연결 중·말소리 감지·듣는 중 > 멈춤.
  const status = state.permission ? '마이크와 동시통역 연결 준비 중 · 처음에는 잠시 걸릴 수 있습니다'
    : state.playing ? '재생 중 · 듣기 멈춤'
    : state.active ? !state.ready ? '다시 연결 중 · 듣기 멈춤' : state.speaking ? '말소리 감지 · 자막 갱신 중' : '듣는 중' : '멈춤'

  return <section className="conversation-panel" aria-labelledby="live-title">
    <h2 id="live-title">동시통역</h2>
    <p id="live-hint" className="hint">{direction.source_lang === 'ko' ? '한국어' : '영어'}로 말하면 원문과 번역문이 자막으로 나타납니다. 약 1초 쉬면 최종 번역과 음성이 나옵니다. 음성 재생 중에는 듣기를 멈춥니다.</p>
    <p className="hint">진한 글자는 확정된 부분, 흐린 글자는 갱신 중인 부분입니다. 원문이 늘면 번역이 바뀔 수 있어 번역문은 최종 결과까지 흐리게 표시합니다.</p>
    {/* 시작·멈춤 토글. 예시 모드(api.demo)에서는 끈다. 멈춘 뒤 다시 듣기로 재생 중일 때만 '음성 멈춤'이 */}
    {/* 나온다. */}
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
    {/* 스크린리더 전용 알림 영역. 최종 결과가 나올 때 세션이 만든 '몇 번째 번역 완료. 원문… 번역문…' */}
    {/* 문장을 한 번 읽는다. */}
    <div className="sr-only" role="status" aria-live="polite" aria-atomic="true">
      {state.announcement}
    </div>
    {/* 마디 목록. 자막이 계속 바뀌므로 aria-live="off"로 바뀔 때마다 읽지 않는다. 대화 모드와 같은 CSS */}
    {/* 클래스를 써서 모양을 맞춘다. */}
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
        {/* 목록은 자동으로 읽지 않으므로 마디 오류와 재생 오류는 role="alert"로 바로 알린다. */}
        {turn.error && <p role="alert" className="error">{turn.error} 다음 말은 계속 번역합니다.</p>}
        {turn.audioError && <p role="alert" className="error">{turn.audioError}</p>}
      </li>)}
    </ol>
    {!state.turns.length && <p className="hint">듣는 중 표시가 나오면 말해 주세요.</p>}
  </section>
}

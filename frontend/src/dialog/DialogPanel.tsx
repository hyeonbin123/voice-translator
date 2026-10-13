// [파일 역할] 번역 화면의 '두 사람 대화' 입력 방식 화면. 두 가지 보기를 제공한다: 말풍선 목록(노트북·모니터,
// 한국어 화자는 왼쪽·영어 화자는 오른쪽)과, 휴대폰을 두 사람 사이에 놓고 쓰는 마주 보기(화면을 위아래로
// 나누고 맞은편 칸을 180도 돌려 각자 자기 언어 번역을 크게 봄, f3f6afb).
// [설계 이유] 흐름은 DialogSession(session.ts)이 관리하고, 이 컴포넌트는 상태를 그리고 시작·멈춤·보기
// 전환·방향 바꾸기·다시 듣기 버튼만 연결한다.
// [연결] translation/TranslatePage.tsx가 입력 방식으로 두 사람 대화를 고르면 그린다. 방향은 고르지
// 않는다(서버가 마디마다 판별).

import { useEffect, useMemo, useState, useSyncExternalStore } from 'react'
import type { Language, TranslationApi } from '../translation/api'
import { DialogSession, type DialogTurn, wasLanguageGuessed } from './session'

// 언어 코드와 마디 상태의 화면 이름.
const languageName = { ko: '한국어', en: '영어' }
const turnStatus = { queued: '번역 대기', translating: '번역 중', done: '번역 완료', error: '번역 실패', canceled: '멈춤 · 전송 취소' }

// 결과가 나온 마디 아래에 붙는 버튼·안내 묶음(말풍선과 마주 보기 칸이 함께 쓴다).
// - 서버 음성이 없으면 브라우저 음성으로 읽는다는 안내
// - 확신이 낮아 방향을 추정한 마디면 그 사실과 '방향 바꾸기' 버튼. 녹음이 이미 지워진(최근 5마디 밖) 마디는
//   버튼을 끄고 이유를 보인다
// - 번역 음성 다시 듣기, 방향 바꾸기 실패와 재생 실패 안내
// busy: 그 마디를 재생하거나 방향을 바꾸는 중이면 버튼을 잠근다.
function ResultActions({ turn, session, busy }: { turn: DialogTurn; session: DialogSession; busy: boolean }) {
  if (!turn.result) return null
  const canReverse = wasLanguageGuessed(turn.result)
  const recording = session.hasRecording(turn.id)
  return <>
    {!turn.result.audio_id && <p className="hint">서버 음성이 없어 브라우저 음성으로 읽습니다.</p>}
    {canReverse && <div className="guess-notice">
      <p>짧거나 애매한 말이라 번갈아 말한 것으로 보고 방향을 추정했습니다.</p>
      <button type="button" disabled={busy || !recording} onClick={() => { void session.reverse(turn.id) }}>
        {busy ? '방향 바꾸는 중…' : recording ? '이 말의 방향 바꾸기' : '방향 바꾸기 만료'}
      </button>
      {!recording && <p className="hint">최근 몇 마디만 녹음을 기억하므로 이 말은 다시 처리할 수 없습니다.</p>}
    </div>}
    <button type="button" disabled={busy} onClick={() => session.replay(turn.id)}
      aria-label={`${turn.id}번째 번역 음성 다시 듣기`}>{busy ? '음성 준비·재생 중' : '번역 음성 다시 듣기'}</button>
    {turn.correctionError && <p role="alert" className="error">{turn.correctionError}</p>}
    {turn.audioError && <p className="error">{turn.audioError}</p>}
  </>
}

// 마주 보기의 한 칸. 그 칸 화자의 언어로 번역된 가장 최근 마디를 크게 보이고, 상대방 원문을 작게 덧붙인다.
// 맞은편 칸(top)은 CSS에서 내용을 180도 돌린다(styles.css의 face-pane-top). aria-label로 어느 쪽 칸인지
// 스크린리더에 알린다.
function FacePane({ language, turn, top, session, busy }: {
  language: Language; turn?: DialogTurn; top: boolean; session: DialogSession; busy: boolean
}) {
  return <section className={`face-pane${top ? ' face-pane-top' : ''}`}
    aria-label={`${languageName[language]} 화자 쪽${top ? ' · 맞은편' : ' · 가까운 쪽'}`}>
    <div className="face-pane-content">
      <h3>{languageName[language]} 화자 쪽</h3>
      {turn?.result ? <>
        <p className="face-translation" lang={language}>{turn.result.translated_text}</p>
        <p className="face-source" lang={turn.result.source_lang}>상대방 원문: {turn.result.source_text}</p>
        <ResultActions turn={turn} session={session} busy={busy} />
      </> : <p className="hint">상대방이 말하면 {languageName[language]} 번역이 여기에 크게 표시됩니다.</p>}
    </div>
  </section>
}

// 상태: 세션 상태(구독), 마주 보기 여부, 맞은편 칸의 언어(기본 한국어). 세션은 api가 바뀔 때만 새로 만들고
// 화면을 떠나면 멈춘다. latestFor는 그 언어로 번역된 가장 최근 마디를 찾는다(마주 보기 칸에 보일 것).
export default function DialogPanel({ api, onActiveChange }: {
  api: TranslationApi; onActiveChange: (active: boolean) => void
}) {
  const session = useMemo(() => new DialogSession(api), [api])
  const state = useSyncExternalStore(session.subscribe, session.getSnapshot)
  const [faceToFace, setFaceToFace] = useState(false)
  const [topLanguage, setTopLanguage] = useState<Language>('ko')
  useEffect(() => () => { session.stop() }, [session])
  useEffect(() => { onActiveChange(state.active) }, [state.active, onActiveChange])
  const status = state.permission ? '마이크와 말소리 감지 준비 중 · 처음에는 잠시 걸릴 수 있습니다'
    : state.playing ? '재생 중 · 듣기 멈춤'
    : state.active ? `${state.speaking ? '말소리 감지' : '듣는 중'}${state.translating ? ' · 번역 중' : ''}` : '멈춤'
  const latestFor = (language: Language) => [...state.turns].reverse().find((turn) => turn.result?.target_lang === language)
  const bottomLanguage = topLanguage === 'ko' ? 'en' : 'ko'

  return <section className="conversation-panel dialog-panel" aria-labelledby="dialog-title">
    <h2 id="dialog-title">두 사람 대화</h2>
    <p id="dialog-hint" className="hint">한국어와 영어로 번갈아 말해 주세요. 언어를 자동으로 가려 상대 언어로 번역하고, 번역 음성이 끝나면 다시 듣습니다.</p>
    <p className="hint">약 1초 동안 말소리가 없으면 한 마디를 보냅니다. 녹음은 저장하지 않으며 방향 수정용으로 최근 몇 마디만 메모리에 둡니다.</p>
    {/* 버튼: 시작·멈춤 토글, 멈춘 뒤 다시 듣기 중일 때의 '음성 멈춤', 말풍선 ↔ 마주 보기 */}
    {/* 전환(aria-pressed), 마주 보기일 때 위아래 칸 언어 바꾸기(휴대폰을 어느 쪽으로 놓았는지에 맞춤). */}
    <div className="actions">
      <button type="button" className="primary" aria-describedby="dialog-hint" aria-pressed={state.active}
        onClick={() => { if (state.active) session.stop(); else void session.start() }}>
        {state.active ? '두 사람 대화 멈춤' : '두 사람 대화 시작'}
      </button>
      {!state.active && state.playingId !== null && <button type="button" onClick={session.stop}>음성 멈춤</button>}
      <button type="button" aria-pressed={faceToFace} onClick={() => setFaceToFace((value) => !value)}>
        {faceToFace ? '말풍선 보기' : '마주 보기'}
      </button>
      {faceToFace && <button type="button" onClick={() => setTopLanguage(bottomLanguage)}>칸 언어 바꾸기</button>}
    </div>
    <p role="status" aria-atomic="true">{status}</p>
    {state.notice && <p role="status" className="hint">{state.notice}</p>}
    {state.error && <p role="alert" className="error">{state.error}</p>}
    <p className="hint">멈추면 아직 끝나지 않은 말과 대기 중인 번역·재생을 취소합니다. 다시 시작한 첫 마디에는 이전 언어를 사용하지 않습니다.</p>
    {/* 보기 전환. 마주 보기: 맞은편 칸과 가까운 칸. 말풍선: 원문 언어에 따라 왼쪽(한국어)·오른쪽(영어) */}
    {/* 정렬 클래스를 주고, 새 말풍선과 글 변화를 스크린리더가 말을 끊지 않고 읽게 */}
    {/* 한다(aria-live="polite"). */}
    {faceToFace ? <div className="face-layout" aria-label="마주 보기 대화" aria-live="polite">
      <FacePane top language={topLanguage} turn={latestFor(topLanguage)} session={session}
        busy={state.playingId === latestFor(topLanguage)?.id || state.correctingId === latestFor(topLanguage)?.id} />
      <FacePane top={false} language={bottomLanguage} turn={latestFor(bottomLanguage)} session={session}
        busy={state.playingId === latestFor(bottomLanguage)?.id || state.correctingId === latestFor(bottomLanguage)?.id} />
    </div> : <ol className="dialog-turns" aria-label="두 사람 대화 내용" aria-live="polite" aria-relevant="additions text">
      {state.turns.map((turn) => <li key={turn.id} className={turn.result?.source_lang === 'en' ? 'dialog-turn-en' : 'dialog-turn-ko'}>
        <article className="dialog-bubble" aria-label={`${turn.id}번째 말 · ${turnStatus[turn.state]}`}>
          <h3>{turn.result ? `${languageName[turn.result.source_lang]} 화자` : `${turn.id}번째 말`} · {turnStatus[turn.state]}</h3>
          {turn.result && <>
            <p className="dialog-source" lang={turn.result.source_lang}>{turn.result.source_text}</p>
            <p className="dialog-translation" lang={turn.result.target_lang}>{turn.result.translated_text}</p>
            <ResultActions turn={turn} session={session}
              busy={state.playingId === turn.id || state.correctingId === turn.id} />
          </>}
          {turn.error && <p className="error">{turn.error} 다음 말은 계속 번역합니다.</p>}
        </article>
      </li>)}
    </ol>}
    {!state.turns.length && !faceToFace && <p className="hint">대화를 시작하면 한국어는 왼쪽, 영어는 오른쪽 말풍선에 쌓입니다.</p>}
  </section>
}

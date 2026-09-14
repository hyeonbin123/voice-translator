import { useEffect, useMemo, useState, useSyncExternalStore } from 'react'
import type { Language, TranslationApi } from '../translation/api'
import { DialogSession, type DialogTurn, wasLanguageGuessed } from './session'

const languageName = { ko: '한국어', en: '영어' }
const turnStatus = { queued: '번역 대기', translating: '번역 중', done: '번역 완료', error: '번역 실패', canceled: '멈춤 · 전송 취소' }

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
    {turn.correctionError && <p role="alert" className="error">{turn.correctionError} 원래 말풍선을 유지했습니다.</p>}
    {turn.audioError && <p className="error">{turn.audioError}</p>}
  </>
}

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

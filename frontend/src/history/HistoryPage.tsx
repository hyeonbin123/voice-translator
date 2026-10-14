// [파일 역할] 번역 기록 화면. /history는 페이지 단위 목록(최신순 20개씩), /history/:id는 기록 상세(원문,
// 번역문, 번역 음성, 단계별 시간, 삭제)를 보인다. 로그인한 본인 기록만 서버가 돌려준다(docs/api.md 번역 기록
// 절).
// [설계 이유] 페이지 위치를 주소의 ?offset=에 둬서 새로 고침·뒤로 가기·링크 공유로 같은 페이지가 열린다.
// 상세로 갈 때도 offset을 실어 '기록 목록으로'가 보던 페이지로 돌아간다. 목록과 상세 컴포넌트에 key를 줘
// 주소가 바뀌면 상태를 새로 시작한다.
// [연결] App.tsx가 두 경로 모두에 이 컴포넌트를 HistoryApi와 함께 그린다. 음성 재생은
// translation/Playback.tsx를 함께 쓴다.

import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { ApiError, errorMessage } from '../api/client'
import Playback from '../translation/Playback'
import { HistoryApi, PAGE_SIZE, type HistoryItem, type HistoryList } from './api'

// 작은 도우미: 언어 이름, 목록 주소 만들기, 날짜를 한국어 형식으로, 오류 문구 고르기(404는 '기록이 없거나
// 삭제됨', 나머지는 공통 오류 문구).
const language = (lang: string) => lang === 'ko' ? '한국어' : '영어'
const listPath = (offset: number) => `/history?offset=${offset}`
const date = (value: string) => new Date(value).toLocaleString('ko-KR')
const historyError = (cause: unknown) => cause instanceof ApiError && cause.status === 404
  ? '기록이 없거나 삭제되었습니다. 목록에서 다시 확인해 주세요.' : errorMessage(cause)

// 주소를 해석해 목록 또는 상세를 고른다. offset은 숫자만 받아들이고, 안전한 정수가 아니거나 형식이 틀리면 0,
// 맞으면 페이지 크기(20)의 배수로 내려 맞춘다. 주소를 손으로 고쳐도 서버에 잘못된 값(422)을 보내지 않는다.
export default function HistoryPage({ api }: { api: HistoryApi }) {
  const { id } = useParams()
  const [search] = useSearchParams()
  const raw = search.get('offset') ?? '0'
  const number = /^\d+$/.test(raw) ? Number(raw) : 0
  const offset = Number.isSafeInteger(number) ? Math.floor(number / PAGE_SIZE) * PAGE_SIZE : 0
  return id ? <HistoryDetail key={id} id={id} offset={offset} api={api} />
    : <HistoryListing key={offset} offset={offset} api={api} />
}

// 목록 한 페이지. offset이나 다시 불러오기(retry)가 바뀌면 다시 받고, 페이지를 빨리 넘기면 앞 요청을
// 취소한다. 기록이 줄어 지금 페이지가 비면(다른 탭이나 상세 화면에서 삭제) 마지막 페이지로 주소를
// 바꾼다(replace라 뒤로 가기 기록을 남기지 않음).
function HistoryListing({ api, offset }: { api: HistoryApi; offset: number }) {
  const [data, setData] = useState<HistoryList | null>(null)
  const [error, setError] = useState('')
  const [retry, setRetry] = useState(0)
  const navigate = useNavigate()
  useEffect(() => {
    const controller = new AbortController()
    void api.list(offset, controller.signal).then((next) => {
      if (controller.signal.aborted) return
      // 다른 탭이나 상세 화면이 이 쪽의 마지막 항목을 지웠을 수 있다.
      if (offset > 0 && offset >= next.total) {
        const last = Math.max(0, Math.floor((next.total - 1) / PAGE_SIZE) * PAGE_SIZE)
        void navigate(listPath(last), { replace: true })
        return
      }
      setData(next)
    }).catch((cause: unknown) => {
      if (!controller.signal.aborted) setError(historyError(cause))
    })
    return () => controller.abort()
  }, [api, offset, retry, navigate])

  return <section className="history-panel">
    <p className="eyebrow">나의 번역</p><h1 tabIndex={-1}>번역 기록</h1>
    {!data && !error && <p role="status">기록을 불러오는 중…</p>}
    {error && <><p className="error" role="alert">{error}</p><button onClick={() => {
      setError(''); setRetry((value) => value + 1)
    }}>기록 다시 불러오기</button></>}
    {/* 전체 개수와 지금 범위(몇~몇 번째), 기록이 없으면 첫 번역으로 가는 링크, 있으면 번호가 이어지는 */}
    {/* 목록(ol start)과 이전·다음 페이지 버튼. 각 항목은 상세로 가는 링크이며, 원문·번역문에 lang을 달아 */}
    {/* 언어별 발음으로 읽힌다. */}
    {data && <>
      <p role="status">전체 {data.total}개{data.total > 0 && ` · ${offset + 1}–${offset + data.items.length}번째`}</p>
      {data.total === 0 ? <p>아직 번역 기록이 없습니다. <Link to="/translate">첫 번역 시작하기</Link></p>
        : <ol className="history-list" start={offset + 1}>
          {data.items.map((item) => <li key={item.id}>
            <Link className="history-link" to={`/history/${encodeURIComponent(item.id)}?offset=${offset}`}>
              <span className="hint">{language(item.source_lang)} → {language(item.target_lang)} · {item.mode === 'speech' ? '음성' : '글자'} · <time dateTime={item.created_at}>{date(item.created_at)}</time></span>
              <span lang={item.source_lang}>{item.source_text}</span>
              <span className="muted" lang={item.target_lang}>{item.translated_text}</span>
            </Link>
          </li>)}
        </ol>}
      <nav aria-label="기록 페이지">
        <button disabled={offset === 0} onClick={() => void navigate(listPath(Math.max(0, offset - PAGE_SIZE)))}>이전 페이지</button>
        <button disabled={offset + PAGE_SIZE >= data.total} onClick={() => void navigate(listPath(offset + PAGE_SIZE))}>다음 페이지</button>
      </nav>
    </>}
  </section>
}

// 기록 상세. id가 바뀌거나 다시 불러오기를 누르면 다시 받는다. 화면을 떠나면 받기와 진행 중인 삭제 요청을
// 모두 취소한다.
function HistoryDetail({ api, id, offset }: { api: HistoryApi; id: string; offset: number }) {
  const [item, setItem] = useState<HistoryItem | null>(null)
  const [error, setError] = useState('')
  const [retry, setRetry] = useState(0)
  const [deleting, setDeleting] = useState(false)
  const deletion = useRef<AbortController | null>(null)
  const navigate = useNavigate()
  useEffect(() => {
    const controller = new AbortController()
    void api.detail(id, controller.signal).then((next) => {
      if (!controller.signal.aborted) setItem(next)
    }).catch((cause: unknown) => {
      if (!controller.signal.aborted) setError(historyError(cause))
    })
    return () => { controller.abort(); deletion.current?.abort() }
  }, [api, id, retry])

  // 삭제. 이미 삭제 중이면 무시하고, 브라우저 확인 창에서 한 번 더 묻는다. 서버는 기록과 연결된 번역 음성
  // 파일도 함께 지운다. 성공하면 보던 목록 페이지로 돌아가고(replace라 뒤로 가기로 지운 상세에 돌아오지
  // 않음), 실패하면 오류를 보이고 다시 누를 수 있게 한다.
  async function remove() {
    if (deletion.current || !window.confirm('이 번역 기록과 연결된 음성을 삭제할까요?')) return
    const controller = new AbortController()
    deletion.current = controller
    setDeleting(true); setError('')
    try {
      await api.remove(id, controller.signal)
      if (!controller.signal.aborted) void navigate(listPath(offset), { replace: true })
    } catch (cause) {
      if (!controller.signal.aborted) {
        setError(historyError(cause)); setDeleting(false); deletion.current = null
      }
    }
  }

  return <section className="history-panel">
    <Link to={listPath(offset)}>기록 목록으로</Link><h1 tabIndex={-1}>기록 상세</h1>
    {!item && !error && <p role="status">기록을 불러오는 중…</p>}
    {error && <p className="error" role="alert">{error}</p>}
    {!item && error && <button onClick={() => { setError(''); setRetry((value) => value + 1) }}>기록 다시 불러오기</button>}
    {/* 방향·입력 방식·시각, 원문·번역문, 번역 음성(서버 음성이 없으면 브라우저 음성), 단계별 모델 처리 */}
    {/* 시간(실행하지 않은 단계는 '실행 안 함'), 삭제 버튼. */}
    {item && <>
      <p className="hint">{language(item.source_lang)} → {language(item.target_lang)} · {item.mode === 'speech' ? '음성' : '글자'} · <time dateTime={item.created_at}>{date(item.created_at)}</time></p>
      <div className="result-columns">
        <div><h2>원문</h2><p lang={item.source_lang}>{item.source_text}</p></div>
        <div><h2>번역문</h2><p lang={item.target_lang}>{item.translated_text}</p></div>
      </div>
      <Playback key={item.id} result={item} api={api} />
      <dl className="timings">
        {([['음성 인식', item.stt_ms], ['번역', item.mt_ms], ['음성 합성', item.tts_ms]] as const).map(([label, ms]) =>
          <div key={label}><dt>{label}</dt><dd>{ms === null ? '실행 안 함' : `${ms} ms`}</dd></div>)}
      </dl>
      <div className="actions"><button className="danger" disabled={deleting} onClick={() => void remove()}>{deleting ? '삭제 중…' : '기록 삭제'}</button></div>
    </>}
  </section>
}

import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { ApiError, errorMessage } from '../api/client'
import Playback from '../translation/Playback'
import { HistoryApi, PAGE_SIZE, type HistoryItem, type HistoryList } from './api'

const language = (lang: string) => lang === 'ko' ? '한국어' : '영어'
const listPath = (offset: number) => `/history?offset=${offset}`
const date = (value: string) => new Date(value).toLocaleString('ko-KR')
const historyError = (cause: unknown) => cause instanceof ApiError && cause.status === 404
  ? '기록이 없거나 삭제되었습니다. 목록에서 다시 확인해 주세요.' : errorMessage(cause)

export default function HistoryPage({ api }: { api: HistoryApi }) {
  const { id } = useParams()
  const [search] = useSearchParams()
  const raw = search.get('offset') ?? '0'
  const number = /^\d+$/.test(raw) ? Number(raw) : 0
  const offset = Number.isSafeInteger(number) ? Math.floor(number / PAGE_SIZE) * PAGE_SIZE : 0
  return id ? <HistoryDetail key={id} id={id} offset={offset} api={api} />
    : <HistoryListing key={offset} offset={offset} api={api} />
}

function HistoryListing({ api, offset }: { api: HistoryApi; offset: number }) {
  const [data, setData] = useState<HistoryList | null>(null)
  const [error, setError] = useState('')
  const [retry, setRetry] = useState(0)
  const navigate = useNavigate()
  useEffect(() => {
    const controller = new AbortController()
    void api.list(offset, controller.signal).then((next) => {
      if (controller.signal.aborted) return
      // Another tab or the detail page may have removed the last item on this page.
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
    <p className="eyebrow">나의 번역</p><h1>번역 기록</h1>
    {!data && !error && <p role="status">기록을 불러오는 중…</p>}
    {error && <><p className="error" role="alert">{error}</p><button onClick={() => {
      setError(''); setRetry((value) => value + 1)
    }}>기록 다시 불러오기</button></>}
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
    <Link to={listPath(offset)}>기록 목록으로</Link><h1>기록 상세</h1>
    {!item && !error && <p role="status">기록을 불러오는 중…</p>}
    {error && <p className="error" role="alert">{error}</p>}
    {!item && error && <button onClick={() => { setError(''); setRetry((value) => value + 1) }}>기록 다시 불러오기</button>}
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

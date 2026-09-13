import { useEffect, useMemo } from 'react'
import { BrowserRouter, Link, Navigate, NavLink, Outlet, Route, Routes, useLocation } from 'react-router-dom'
import { api } from './api/client'
import type { ApiClient } from './api/client'
import AuthPage from './auth/AuthPage'
import { useSession } from './auth/useSession'
import TranslatePage from './translation/TranslatePage'
import { TranslationApi } from './translation/api'
import { mockTranslationApi } from './translation/mock'

function ProtectedLayout({ client }: { client: ApiClient }) {
  const session = useSession(client)
  const location = useLocation()
  if (session.status !== 'authenticated') {
    return <Navigate to="/login" state={{ from: location.pathname }} replace />
  }
  return (
    <>
      <div className="workspace-bar">
        <nav aria-label="주 메뉴">
          <NavLink to="/translate">번역</NavLink>
          <NavLink to="/history">기록</NavLink>
        </nav>
        <div className="account"><span>{session.user?.email}</span><button onClick={() => client.logout()}>로그아웃</button></div>
      </div>
      <Outlet />
    </>
  )
}

function Placeholder({ history = false }: { history?: boolean }) {
  return (
    <section className="placeholder">
      <p className="eyebrow">{history ? '나의 번역' : '영어 ↔ 한국어'}</p>
      <h1>{history ? '번역 기록' : '번역'}</h1>
      <p>{history ? '번역 기록을 모아 볼 공간입니다.' : '말하거나 입력한 내용을 번역할 공간입니다.'}</p>
      <p className="muted">{history ? '기록 조회 기능을 준비하고 있습니다.' : '음성·텍스트 번역 기능을 준비하고 있습니다.'}</p>
    </section>
  )
}

export function AppRoutes({ client = api }: { client?: ApiClient }) {
  const session = useSession(client)
  const translation = useMemo(() => import.meta.env.DEV && import.meta.env.VITE_TRANSLATION_MOCK !== 'false'
    ? mockTranslationApi : new TranslationApi(client), [client])
  useEffect(() => { void client.initialize() }, [client])

  return (
    <div className="app">
      <header className="brand"><Link to="/">voice-translator</Link><span>말과 글로 이어지는 대화</span></header>
      <main id="main">
        {session.status === 'loading' ? <p role="status">로그인 확인 중…</p> : (
          <Routes>
            <Route path="/login" element={<AuthPage key="login" client={client} />} />
            <Route path="/register" element={<AuthPage key="register" client={client} register />} />
            <Route element={<ProtectedLayout client={client} />}>
              <Route path="/translate" element={<TranslatePage api={translation} />} />
              <Route path="/history" element={<Placeholder history />} />
            </Route>
            <Route path="/" element={<Navigate to="/translate" replace />} />
            <Route path="*" element={<section className="placeholder"><h1>페이지를 찾을 수 없습니다</h1><Link to="/">처음으로</Link></section>} />
          </Routes>
        )}
      </main>
    </div>
  )
}

export default function App() {
  return <BrowserRouter><AppRoutes /></BrowserRouter>
}

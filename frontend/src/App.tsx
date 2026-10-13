// [파일 역할] 앱의 뼈대: 주소(라우트)와 화면을 잇고, 로그인 상태에 따라 보호 화면을 막고, 화면 공통
// 머리글·본문 바로가기·제목 포커스 같은 접근성 동작을 맡는다.
// [주요 구성] ProtectedLayout(로그인한 사용자 전용 틀: 메뉴·계정·로그아웃) → AppRoutes(라우트 표, 세션 복원,
// 화면 이동마다 제목 갱신·포커스) → App(실제 브라우저 주소를 쓰는 BrowserRouter로 감쌈).
// [연결] main.tsx가 App을 그린다. 테스트(App.test.tsx)는 AppRoutes를 MemoryRouter와 가짜 fetch를 쓰는
// ApiClient로 감싸 주소와 서버 응답을 마음대로 정한다.

// react-router-dom: 주소별 화면 전환(Routes·Route), 보호 화면에서 로그인으로 보내기(Navigate), 현재 메뉴
// 표시(NavLink). 프로젝트 모듈: 로그인 상태를 가진 api 클라이언트, 로그인·가입 화면, 번역 화면과 그
// API(실제/예시), 기록 화면과 그 API.
import { useEffect, useMemo, useRef } from 'react'
import { BrowserRouter, Link, Navigate, NavLink, Outlet, Route, Routes, useLocation } from 'react-router-dom'
import { api } from './api/client'
import type { ApiClient } from './api/client'
import AuthPage from './auth/AuthPage'
import { useSession } from './auth/useSession'
import TranslatePage from './translation/TranslatePage'
import { TranslationApi } from './translation/api'
import { mockTranslationApi } from './translation/mock'
import HistoryPage from './history/HistoryPage'
import { HistoryApi } from './history/api'

// 로그인한 사용자만 볼 수 있는 화면(번역, 기록)의 공통 틀. 자식 라우트는 <Outlet /> 자리에 그려진다.
// 로그인 상태가 아니면 /login으로 보낸다. 이때 원래 가려던 경로를 state.from에 실어 로그인 뒤 그 화면(기록
// 또는 번역)으로 돌아가게 하고, replace로 기록(history)을 바꿔 뒤로 가기가 다시 보호 화면 → 로그인으로 튕기는
// 고리를 만들지 않는다.
// 세션 복원 중(loading)에는 AppRoutes가 라우트 자체를 그리지 않으므로, 이 검사는 복원이 끝난 뒤에만 돈다.
function ProtectedLayout({ client }: { client: ApiClient }) {
  const session = useSession(client)
  const location = useLocation()
  if (session.status !== 'authenticated') {
    return <Navigate to="/login" state={{ from: location.pathname }} replace />
  }
  return (
    <>
      {/* 상단 줄: 번역·기록 메뉴와 로그인한 이메일, 로그아웃 버튼. NavLink는 현재 경로의 링크에 active */}
      {/* 표시와 aria-current를 자동으로 붙인다. 로그아웃은 토큰을 지우고 세션을 anonymous로 바꾸며, */}
      {/* 그러면 이 틀이 다시 그려지면서 /login으로 이동한다. */}
      <div className="workspace-bar">
        <nav aria-label="주 메뉴">
          <NavLink to="/translate">번역</NavLink>
          <NavLink to="/history">기록</NavLink>
        </nav>
        <div className="account"><span>{session.user?.email}</span><button onClick={() => client.logout()}>로그아웃</button></div>
      </div>
      {/* 자식 라우트(번역 화면, 기록 화면)가 그려지는 자리. */}
      <Outlet />
    </>
  )
}

// 라우트 표와 화면 공통 동작. client를 인자로 받는 이유는 테스트가 가짜 저장소·fetch를 쓰는 ApiClient를 넣을
// 수 있게 하려는 것이다(기본값은 앱 전체가 쓰는 api 하나).
export function AppRoutes({ client = api }: { client?: ApiClient }) {
  const session = useSession(client)
  const location = useLocation()
  // main: 본문 영역. 화면 이동 뒤 그 안의 h1을 찾아 포커스를 옮기는 데 쓴다. previousRoute: 직전 경로. 처음
  // 그릴 때와 같은 경로에서 다시 그릴 때는 포커스를 옮기지 않으려고 기억한다.
  const main = useRef<HTMLElement>(null)
  const previousRoute = useRef(location.pathname + location.search)
  // 번역 API 고르기: 빌드 환경 변수 VITE_TRANSLATION_MOCK가 정확히 'true'일 때만 서버 없이 정해진 예시 문장만
  // 번역하는 예시 모드(translation/mock.ts)를 쓰고, 그 밖에는 실제 서버 API를 쓴다. 모델 없이 화면만 볼 때를
  // 위한 설정이다(README 로컬 개발). 배포 이미지는 Dockerfile에서 이 값을 false로 고정한다.
  // useMemo로 client가 바뀔 때만 새로 만들어, 렌더마다 API 객체가 바뀌어 하위 화면의 effect가 다시 도는 일을
  // 막는다.
  const translation = useMemo(() => import.meta.env.VITE_TRANSLATION_MOCK === 'true'
    ? mockTranslationApi : new TranslationApi(client), [client])
  const history = useMemo(() => new HistoryApi(client), [client])
  // 처음 그릴 때 세션을 복원한다: sessionStorage에 refresh 토큰이 있으면 새 access 토큰을 받아 /api/auth/me로
  // 사용자를 확인하고, 없으면 바로 로그아웃 상태가 된다. initialize는 진행 중인 약속(Promise)을 기억하므로
  // StrictMode가 effect를 두 번 불러도 갱신 요청은 한 번만 나간다(App.test.tsx가 확인).
  useEffect(() => { void client.initialize() }, [client])
  // 화면이 바뀔 때마다: (1) 문서 제목을 '화면 h1 · voice-translator'로 바꿔 탭과 스크린리더가 현재 화면을
  // 알게 하고, (2) 경로가 실제로 바뀌었으면 새 화면의 h1로 포커스를 옮긴다. 단일 페이지 앱은 화면이 바뀌어도
  // 포커스가 이전 위치(누른 링크)에 남아, 키보드·스크린리더 사용자가 새 내용이 나온 것을 알기 어렵기
  // 때문이다(faf1e55). 세션 복원 중에는 아직 화면이 없으므로 건너뛴다.
  useEffect(() => {
    if (session.status === 'loading') return
    const heading = main.current?.querySelector('h1')
    if (heading) document.title = `${heading.textContent} · voice-translator`
    const route = location.pathname + location.search
    if (route !== previousRoute.current) heading?.focus()
    previousRoute.current = route
  }, [location.pathname, location.search, session.status])

  return (
    <div className="app">
      {/* 본문 바로가기 링크(키보드 포커스를 받을 때만 보임, styles.css). 기본 이동을 막고 본문의 */}
      {/* h1(없으면 main)에 직접 포커스를 준다. #main으로 주소를 바꾸지 않아 라우터 주소가 흐트러지지 */}
      {/* 않고, 포커스가 확실히 본문으로 간다. */}
      <a className="skip-link" href="#main" onClick={(event) => {
        event.preventDefault()
        const destination = main.current?.querySelector('h1') ?? main.current
        destination?.focus()
      }}>본문 바로가기</a>
      {/* 모든 화면에 보이는 머리글. 서비스 이름을 누르면 /로 가고, /는 /translate로 넘어간다. */}
      <header className="brand"><Link to="/">voice-translator</Link><span>말과 글로 이어지는 대화</span></header>
      {/* tabIndex={-1}: Tab 순서에는 넣지 않되 코드로 포커스를 줄 수 있게 한다(바로가기 링크의 대상). */}
      <main id="main" ref={main} tabIndex={-1}>
        {/* 세션 복원이 끝날 때까지는 라우트를 그리지 않고 상태 문구만 보인다. 그러지 않으면 복원 전 잠깐 */}
        {/* anonymous로 보여 로그인한 사용자도 /login으로 튕기게 된다. */}
        {session.status === 'loading' ? <p role="status">로그인 확인 중…</p> : (
          // 라우트 표: 로그인·가입은 누구나, 번역·기록·기록 상세는 ProtectedLayout 안(로그인 필요), /는
          // 번역으로, 나머지 주소는 '페이지를 찾을 수 없습니다'.
          <Routes>
            {/* 로그인과 가입이 같은 AuthPage를 쓰므로 key를 달리해, 두 화면을 오갈 때 컴포넌트를 새로 */}
            {/* 만들어 입력값·오류 상태가 섞이지 않게 한다. */}
            <Route path="/login" element={<AuthPage key="login" client={client} />} />
            <Route path="/register" element={<AuthPage key="register" client={client} register />} />
            <Route element={<ProtectedLayout client={client} />}>
              <Route path="/translate" element={<TranslatePage api={translation} />} />
              {/* 목록(/history)과 상세(/history/:id)가 같은 HistoryPage다. 화면 안에서 :id가 있으면 */}
              {/* 상세를 보인다. */}
              <Route path="/history" element={<HistoryPage api={history} />} />
              <Route path="/history/:id" element={<HistoryPage api={history} />} />
            </Route>
            <Route path="/" element={<Navigate to="/translate" replace />} />
            <Route path="*" element={<section className="placeholder"><h1 tabIndex={-1}>페이지를 찾을 수 없습니다</h1><Link to="/">처음으로</Link></section>} />
          </Routes>
        )}
      </main>
    </div>
  )
}

// 실제 앱: 브라우저 주소창을 쓰는 BrowserRouter로 AppRoutes를 감싼다. 새로 고침이나 직접 연 주소(/history/...
// 등)는 배포에서 nginx가 index.html로 돌려주고 여기서 처리한다.
export default function App() {
  return <BrowserRouter><AppRoutes /></BrowserRouter>
}

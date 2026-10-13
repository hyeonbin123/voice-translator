// [파일 역할] 로그인 화면과 회원가입 화면. 같은 컴포넌트를 register 값으로 나눠 쓴다(App.tsx의 /login,
// /register 라우트).
// [흐름] 가입: 비밀번호 길이를 미리 검사 → POST /api/auth/register → 로그인 화면으로 이동하며 '가입 완료'
// 안내. 로그인: ApiClient.login이 토큰을 받고 사용자 정보를 확인 → 세션이 authenticated로 바뀌면 이 화면이
// 원래 가려던 화면으로 넘긴다.
// [설계 이유] 서버 오류 문구는 그대로 보이지 않고 api/client.ts의 errorMessage가 상태 코드에 맞는 고정 한국어
// 문구로 바꾼다. 오류가 나면 그 문구로 포커스를 옮겨 키보드·스크린리더 사용자가 바로 알 수 있게 한다.

import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { Link, Navigate, useLocation, useNavigate } from 'react-router-dom'
import { errorMessage } from '../api/client'
import type { ApiClient } from '../api/client'
import { useSession } from './useSession'

// 입력: client(로그인 상태를 가진 ApiClient), register(true면 가입 화면). 상태: 입력한 이메일·비밀번호, 오류
// 문구, 요청 중 여부(busy).
export default function AuthPage({ client, register = false }: { client: ApiClient; register?: boolean }) {
  const session = useSession(client)
  const location = useLocation()
  const navigate = useNavigate()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  // 오류 문구가 생기면 그 문단으로 포커스를 옮긴다. 제출 버튼에 머물던 포커스가 오류 내용으로 가서 무엇이
  // 틀렸는지 바로 읽힌다.
  const errorSummary = useRef<HTMLParagraphElement>(null)
  useEffect(() => { if (error) errorSummary.current?.focus() }, [error])
  // 로그인 뒤 갈 곳. 보호 화면에서 넘어올 때 실린 state.from을 그대로 쓰지 않고, /history일 때만 기록으로,
  // 나머지는 모두 /translate로 정한다. 정해진 두 화면 밖의 임의 경로로는 보내지 않는다.
  const destination = location.state?.from === '/history' ? '/history' : '/translate'

  // 이미 로그인했거나 방금 로그인에 성공해 세션이 바뀌면 이 화면 대신 목적지로 넘어간다. replace라 뒤로
  // 가기로 로그인 화면에 돌아오지 않는다.
  if (session.status === 'authenticated') return <Navigate to={destination} replace />

  // 폼 제출 처리. 기본 제출(페이지 새로 고침)을 막고, 요청 중이면 두 번 보내지 않는다.
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (busy) return
    setError('')
    // 가입 비밀번호를 서버 규칙(8자 이상, UTF-8 72바이트 이하, docs/api.md)과 같게 미리 검사해 헛 요청을
    // 줄인다. [...password]는 UTF-16 단위가 아니라 글자(코드 포인트) 단위로 세고, TextEncoder는 실제 UTF-8
    // 바이트 수를 센다. 72바이트는 서버가 쓰는 bcrypt가 다룰 수 있는 최대 길이라 한글은 24자까지다.
    if (register && ([...password].length < 8 || new TextEncoder().encode(password).length > 72)) {
      setError('비밀번호는 8자 이상, UTF-8 기준 72바이트 이하여야 합니다.')
      return
    }
    setBusy(true)
    try {
      // 가입은 토큰을 주지 않으므로 로그인 화면으로 보내며 '가입 완료' 안내(registered)와 원래 목적지(from)를
      // 넘긴다. 로그인은 성공하면 세션이 바뀌어 위의 Navigate가 처리하므로 여기서 이동하지 않는다. 이메일
      // 앞뒤 공백은 잘라서 보낸다.
      if (register) {
        await client.register(email.trim(), password)
        navigate('/login', { replace: true, state: { registered: true, from: destination } })
      } else {
        await client.login(email.trim(), password)
      }
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      // 성공이든 실패든 비밀번호 입력값을 비운다. 화면 상태에 비밀번호를 남겨 두지 않는다.
      setPassword('')
      setBusy(false)
    }
  }

  return (
    <section className="auth-card" aria-labelledby="auth-title">
      <p className="eyebrow">영어 ↔ 한국어</p>
      <h1 id="auth-title" tabIndex={-1}>{register ? '회원가입' : '로그인'}</h1>
      <p className="muted">{register ? '계정을 만들고 번역 기록을 한곳에 모으세요.' : '로그인하고 나의 번역을 이어가세요.'}</p>
      {/* 안내 문구 두 가지: 방금 가입을 마쳤을 때, 그리고 토큰 갱신이 거절돼 세션이 끝났을 때(expired). */}
      {/* role="status"라 스크린리더가 화면 흐름을 끊지 않고 읽어 준다. */}
      {!register && location.state?.registered && <p role="status" className="notice">가입이 완료되었습니다. 로그인해 주세요.</p>}
      {!register && session.expired && <p role="status" className="notice">로그인이 만료되었습니다. 다시 로그인해 주세요.</p>}
      {/* 요청 중에는 aria-busy와 disabled로 입력과 버튼을 잠근다. 오류가 있으면 각 입력의 */}
      {/* aria-describedby가 오류 문단(auth-error)을 가리켜, 입력란에 포커스가 갈 때 오류도 함께 읽힌다. */}
      <form onSubmit={submit} aria-busy={busy}>
        <label htmlFor="email">이메일</label>
        {/* autoComplete="username": 브라우저 비밀번호 관리자가 이 칸을 계정 이름으로 알아보게 한다(서버의 */}
        {/* 로그인 폼 필드 이름도 username이다). */}
        <input id="email" name="email" type="email" autoComplete="username" required
          aria-describedby={error ? 'auth-error' : undefined}
          value={email} onChange={(event) => setEmail(event.target.value)} disabled={busy} />
        <label htmlFor="password">비밀번호</label>
        {/* 가입 화면은 new-password(새 비밀번호 제안), 로그인 화면은 current-password(저장된 비밀번호 */}
        {/* 채우기)로 비밀번호 관리자의 동작을 나눈다. 가입 화면에서는 조건 안내 문단(password-hint)도 */}
        {/* 함께 읽히게 한다. */}
        <input id="password" name="password" type="password" required
          autoComplete={register ? 'new-password' : 'current-password'}
          aria-describedby={[register ? 'password-hint' : '', error ? 'auth-error' : ''].filter(Boolean).join(' ') || undefined}
          value={password} onChange={(event) => setPassword(event.target.value)} disabled={busy} />
        {register && <p id="password-hint" className="hint">8자 이상, UTF-8 기준 72바이트 이하 (한글은 최대 24자)</p>}
        {/* 오류 문단: role="alert"로 바로 읽히고, tabIndex={-1}이라 위 effect가 포커스를 줄 수 있다. */}
        {error && <p id="auth-error" ref={errorSummary} tabIndex={-1} role="alert" className="error">{error}</p>}
        <button className="primary" type="submit" disabled={busy}>
          {busy ? '처리 중…' : register ? '가입하기' : '로그인'}
        </button>
      </form>
      {/* 로그인 ↔ 가입 전환 링크. 원래 목적지(from)를 그대로 넘겨, 가입을 거쳐도 처음 가려던 화면으로 */}
      {/* 돌아간다. */}
      <p className="switch-auth">{register ? '이미 계정이 있나요? ' : '처음 방문하셨나요? '}
        <Link to={register ? '/login' : '/register'} state={{ from: destination }}>
          {register ? '로그인' : '회원가입'}
        </Link>
      </p>
    </section>
  )
}

import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { Link, Navigate, useLocation, useNavigate } from 'react-router-dom'
import { errorMessage } from '../api/client'
import type { ApiClient } from '../api/client'
import { useSession } from './useSession'

export default function AuthPage({ client, register = false }: { client: ApiClient; register?: boolean }) {
  const session = useSession(client)
  const location = useLocation()
  const navigate = useNavigate()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const errorSummary = useRef<HTMLParagraphElement>(null)
  useEffect(() => { if (error) errorSummary.current?.focus() }, [error])
  const destination = location.state?.from === '/history' ? '/history' : '/translate'

  if (session.status === 'authenticated') return <Navigate to={destination} replace />

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (busy) return
    setError('')
    if (register && ([...password].length < 8 || new TextEncoder().encode(password).length > 72)) {
      setError('비밀번호는 8자 이상, UTF-8 기준 72바이트 이하여야 합니다.')
      return
    }
    setBusy(true)
    try {
      if (register) {
        await client.register(email.trim(), password)
        navigate('/login', { replace: true, state: { registered: true, from: destination } })
      } else {
        await client.login(email.trim(), password)
      }
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setPassword('')
      setBusy(false)
    }
  }

  return (
    <section className="auth-card" aria-labelledby="auth-title">
      <p className="eyebrow">영어 ↔ 한국어</p>
      <h1 id="auth-title" tabIndex={-1}>{register ? '회원가입' : '로그인'}</h1>
      <p className="muted">{register ? '계정을 만들고 번역 기록을 한곳에 모으세요.' : '로그인하고 나의 번역을 이어가세요.'}</p>
      {!register && location.state?.registered && <p role="status" className="notice">가입이 완료되었습니다. 로그인해 주세요.</p>}
      {!register && session.expired && <p role="status" className="notice">로그인이 만료되었습니다. 다시 로그인해 주세요.</p>}
      <form onSubmit={submit} aria-busy={busy}>
        <label htmlFor="email">이메일</label>
        <input id="email" name="email" type="email" autoComplete="username" required
          aria-describedby={error ? 'auth-error' : undefined}
          value={email} onChange={(event) => setEmail(event.target.value)} disabled={busy} />
        <label htmlFor="password">비밀번호</label>
        <input id="password" name="password" type="password" required
          autoComplete={register ? 'new-password' : 'current-password'}
          aria-describedby={[register ? 'password-hint' : '', error ? 'auth-error' : ''].filter(Boolean).join(' ') || undefined}
          value={password} onChange={(event) => setPassword(event.target.value)} disabled={busy} />
        {register && <p id="password-hint" className="hint">8자 이상, UTF-8 기준 72바이트 이하 (한글은 최대 24자)</p>}
        {error && <p id="auth-error" ref={errorSummary} tabIndex={-1} role="alert" className="error">{error}</p>}
        <button className="primary" type="submit" disabled={busy}>
          {busy ? '처리 중…' : register ? '가입하기' : '로그인'}
        </button>
      </form>
      <p className="switch-auth">{register ? '이미 계정이 있나요? ' : '처음 방문하셨나요? '}
        <Link to={register ? '/login' : '/register'} state={{ from: destination }}>
          {register ? '로그인' : '회원가입'}
        </Link>
      </p>
    </section>
  )
}

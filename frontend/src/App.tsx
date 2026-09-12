import { useEffect, useState } from 'react'

type ServerStatus = 'checking' | 'ok' | 'down'

const STATUS_TEXT: Record<ServerStatus, string> = {
  checking: '서버 확인 중',
  ok: '서버 연결됨',
  down: '서버에 연결할 수 없어요',
}

export default function App() {
  const [status, setStatus] = useState<ServerStatus>('checking')

  useEffect(() => {
    let cancelled = false
    fetch('/api/health')
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error(`HTTP ${res.status}`))))
      .then((body: { status?: string }) => {
        if (!cancelled) setStatus(body.status === 'ok' ? 'ok' : 'down')
      })
      .catch(() => {
        if (!cancelled) setStatus('down')
      })
    return () => {
      cancelled = true
    }
  }, [])

  return (
    <main className="app">
      <h1>voice-translator</h1>
      <p>영어↔한국어 음성 번역</p>
      <p role="status">{STATUS_TEXT[status]}</p>
    </main>
  )
}

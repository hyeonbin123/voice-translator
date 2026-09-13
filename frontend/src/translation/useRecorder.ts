import { useEffect, useRef, useState } from 'react'
import { MAX_AUDIO_BYTES } from './api'

export const MAX_RECORDING_MS = 30_000
type Recording = {
  recorder: MediaRecorder
  stream: MediaStream
  timer: ReturnType<typeof setTimeout>
  ticker: ReturnType<typeof setInterval>
}

export function useRecorder(onRecorded: (blob: Blob) => void) {
  const [state, setState] = useState<'idle' | 'permission' | 'recording'>('idle')
  const [seconds, setSeconds] = useState(0)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const generation = useRef(0)
  const active = useRef<Recording | null>(null)

  function release() {
    const current = active.current
    active.current = null
    if (!current) return
    clearTimeout(current.timer)
    clearInterval(current.ticker)
    current.stream.getTracks().forEach((track) => track.stop())
    current.recorder.ondataavailable = null
    current.recorder.onstop = null
    current.recorder.onerror = null
    if (current.recorder.state !== 'inactive') current.recorder.stop()
  }

  useEffect(() => () => {
    generation.current += 1
    release()
  }, [])

  function cancel() {
    generation.current += 1
    release()
    setState('idle')
    setSeconds(0)
    setNotice('녹음을 취소했습니다. 음성은 전송하지 않았습니다.')
  }

  function stop() {
    const current = active.current
    if (current?.recorder.state === 'recording') current.recorder.stop()
  }

  async function start() {
    if (active.current || state !== 'idle') return
    setError('')
    setNotice('')
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      setError('이 브라우저에서는 녹음을 사용할 수 없습니다. Chrome·Edge의 HTTPS 또는 localhost에서 열거나 음성 파일을 선택해 주세요.')
      return
    }
    const ticket = ++generation.current
    setState('permission')
    let stream: MediaStream | undefined
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      if (ticket !== generation.current) {
        stream.getTracks().forEach((track) => track.stop())
        return
      }
      const mimeType = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/mp4']
        .find((type) => MediaRecorder.isTypeSupported(type))
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)
      const chunks: Blob[] = []
      let bytes = 0
      recorder.ondataavailable = (event) => {
        if (ticket !== generation.current) return
        bytes += event.data.size
        if (bytes > MAX_AUDIO_BYTES) {
          generation.current += 1
          release()
          setState('idle')
          setError('녹음 파일이 10MB를 넘었습니다. 더 짧게 녹음해 주세요.')
          return
        }
        if (event.data.size) chunks.push(event.data)
      }
      recorder.onerror = () => {
        if (ticket !== generation.current) return
        generation.current += 1
        release()
        setState('idle')
        setError('녹음 중 오류가 발생했습니다. 마이크를 확인하고 다시 시도해 주세요.')
      }
      recorder.onstop = () => {
        if (ticket !== generation.current) return
        const blob = new Blob(chunks, { type: recorder.mimeType || chunks[0]?.type || 'audio/webm' })
        release()
        setState('idle')
        if (!blob.size) setError('녹음된 음성이 없습니다. 다시 녹음해 주세요.')
        else onRecorded(blob)
      }
      const started = Date.now()
      active.current = {
        recorder, stream,
        timer: setTimeout(() => {
          setNotice('30초 제한에 도달해 녹음을 종료했습니다.')
          stop()
        }, MAX_RECORDING_MS),
        ticker: setInterval(() => setSeconds(Math.min(30, Math.floor((Date.now() - started) / 1000))), 250),
      }
      recorder.start(250)
      setSeconds(0)
      setState('recording')
    } catch (cause) {
      stream?.getTracks().forEach((track) => track.stop())
      if (ticket !== generation.current) return
      release()
      setState('idle')
      setError(cause instanceof DOMException && cause.name === 'NotAllowedError'
        ? '마이크 권한이 거부되었습니다. 브라우저에서 권한을 허용하거나 음성 파일을 선택해 주세요.'
        : '마이크를 시작할 수 없습니다. 연결 상태를 확인하거나 음성 파일을 선택해 주세요.')
    }
  }

  return { state, seconds, error, notice, start, stop, cancel }
}

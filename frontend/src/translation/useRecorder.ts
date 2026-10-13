// [파일 역할] 번역 화면 '음성 입력'의 녹음 훅(누르고 말하고 끝내는 방식). 브라우저 MediaRecorder로 압축
// 형식(기본 webm/Opus)을 녹음해 끝나면 Blob 하나를 onRecorded로 넘긴다. 상태(대기·권한 대기·녹음 중), 경과
// 초, 오류·안내 문구를 화면에 준다.
// [설계 이유] 서버 계약(파일 10MB, 음성 30초, docs/api.md)을 녹음 중에도 지킨다: 250ms마다 데이터를 받아
// 크기를 세고, 29초에 스스로 멈춘다. 대화 모드의 마이크(conversation/microphone.ts)와 달리 소리 조각을
// 실시간으로 다룰 필요가 없어 MediaRecorder를 쓴다.
// [연결] TranslatePage.tsx가 쓴다. 녹음 원본은 서버에 저장되지 않는다(README 설계 판단).

import { useEffect, useRef, useState } from 'react'
import { MAX_AUDIO_BYTES } from './api'

// The server rejects more than 30 s of decoded audio. The stop timer only fires late (hidden pages wake
// timers once per second), so stop a second early, like the conversation modes' 29 s limit.
// 위 영어 주석의 요지: 서버는 30초 넘는 음성을 거절하는데, 숨겨진 탭에서는 타이머가 1초 가까이 늦게 깨어날 수
// 있어 30초 타이머로는 30.0~31.0초에 멈춰 422가 났다. 그래서 1초 일찍 29초에 멈춘다(dfc251d). 화면 문구와
// 카운터는 서버 한도인 30초를 그대로 보인다.
export const MAX_RECORDING_MS = 29_000
// 진행 중인 녹음 한 건이 쥐고 있는 자원: 녹음기, 마이크 스트림, 29초 정지 타이머, 경과 초 갱신 타이머.
type Recording = {
  recorder: MediaRecorder
  stream: MediaStream
  timer: ReturnType<typeof setTimeout>
  ticker: ReturnType<typeof setInterval>
}

// 상태와 ref.
// - generation: 시작·취소·실패·언마운트마다 올리는 번호. 각 녹음은 시작할 때 번호(ticket)를 받고, 늦게 도착한
//   권한 허락·데이터·이벤트의 번호가 지금과 다르면 무시한다
// - active: 진행 중인 녹음의 자원(없으면 null)
export function useRecorder(onRecorded: (blob: Blob) => void) {
  const [state, setState] = useState<'idle' | 'permission' | 'recording'>('idle')
  const [seconds, setSeconds] = useState(0)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const generation = useRef(0)
  const active = useRef<Recording | null>(null)

  // 진행 중인 녹음의 자원을 모두 놓는다: 타이머 해제, 마이크 트랙 정지, 처리기 떼기, 녹음기 정지. 처리기를
  // 먼저 떼므로 여기서 멈춘 녹음은 onRecorded로 넘어가지 않는다.
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

  // 화면을 떠나면 번호를 올리고 자원을 놓아 마이크가 켜진 채 남지 않게 한다.
  useEffect(() => () => {
    generation.current += 1
    release()
  }, [])

  // 녹음 취소: 녹음을 버리고 음성을 전송하지 않았다고 알린다.
  function cancel() {
    generation.current += 1
    release()
    setState('idle')
    setSeconds(0)
    setNotice('녹음을 취소했습니다. 음성은 전송하지 않았습니다.')
  }

  // 녹음 끝내기: 녹음기를 멈추면 onstop에서 Blob이 만들어져 넘어간다.
  function stop() {
    const current = active.current
    if (current?.recorder.state === 'recording') current.recorder.stop()
  }

  // 녹음 시작. 브라우저가 녹음을 지원하지 않으면(보안 컨텍스트가 아니거나 MediaRecorder 없음) 파일 선택을
  // 안내한다. 권한 창을 기다리는 동안 상태는 permission이다.
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
      // 권한 창을 기다리는 사이 취소됐으면 받은 마이크를 바로 놓는다.
      if (ticket !== generation.current) {
        stream.getTracks().forEach((track) => track.stop())
        return
      }
      // 브라우저가 지원하는 첫 형식을 고른다. webm/Opus가 서버의 기본 업로드 형식이고(docs/api.md), 나머지는
      // 다른 브라우저용 대안이다. 아무것도 없으면 브라우저 기본값을 쓴다.
      const mimeType = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/mp4']
        .find((type) => MediaRecorder.isTypeSupported(type))
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)
      const chunks: Blob[] = []
      let bytes = 0
      // 데이터가 올 때마다 누적 크기를 세고, 10MB 한도를 넘으면 녹음을 버리고 더 짧게 녹음하라고 알린다.
      // 서버로 보내 413을 받기 전에 막는다.
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
      // 멈추면 모은 조각을 녹음기의 MIME 형식으로 하나로 묶어 넘긴다. 아무것도 녹음되지 않았으면 오류로
      // 알린다.
      recorder.onstop = () => {
        if (ticket !== generation.current) return
        const blob = new Blob(chunks, { type: recorder.mimeType || chunks[0]?.type || 'audio/webm' })
        release()
        setState('idle')
        if (!blob.size) setError('녹음된 음성이 없습니다. 다시 녹음해 주세요.')
        else onRecorded(blob)
      }
      // 29초 자동 정지 타이머와, 화면에 보일 경과 초(최대 30)를 250ms마다 갱신하는 타이머를 건다.
      // recorder.start(250)은 250ms마다 데이터를 내보내 녹음 중 크기 검사가 가능하게 한다.
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
      // 시작 실패: 받은 마이크를 놓고, 권한 거부(NotAllowedError)와 그 밖의 실패를 나눠 안내한다. 두 경우
      // 모두 파일 선택을 대안으로 알려 준다.
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

// [파일 역할] 웹소켓을 쓰는 세션 테스트(live·conversation·dialog의 session.test.ts)가 함께 쓰는 가짜 소켓.
// 실제 서버 없이 연결 열림, 서버 메시지 도착, 닫힘 코드(예: 4401)를 테스트가 원하는 순서로 일으키고,
// 브라우저가 보낸 메시지를 꺼내 확인한다.
// [연결] live/socket.ts의 LiveTransport는 소켓을 여는 함수를 인자로 받으므로, 테스트는 이 클래스를 돌려주는
// 함수를 넘긴다.

import { vi } from 'vitest'
import type { LiveSocket } from '../live/socket'

/** A WebSocket the test drives: open it, deliver server messages, close it with a code. */
export class FakeSocket implements LiveSocket {
  // readyState 0은 연결 중(CONNECTING), 1은 열림(OPEN), 3은 닫힘(CLOSED)이다. bufferedAmount는 0으로 두어
  // 전송 지연 검사에 걸리지 않게 한다.
  readyState: WebSocket['readyState'] = 0; bufferedAmount = 0
  onopen: WebSocket['onopen'] = null
  onmessage: WebSocket['onmessage'] = null
  onclose: WebSocket['onclose'] = null
  onerror: WebSocket['onerror'] = null
  // send·close는 Vitest 가짜 함수라 몇 번, 무엇으로 불렸는지 기록된다.
  send = vi.fn<(data: string | ArrayBufferLike | Blob | ArrayBufferView) => void>()
  close = vi.fn(() => { this.readyState = 3 })
  // 테스트가 부르는 조작 함수: opened는 연결 열림, message는 서버 JSON 메시지 도착(실제처럼 문자열로 바꿔
  // 넘김), closed는 주어진 코드로 닫힘. 실제 WebSocket처럼 처리기 안의 this가 소켓이 되도록 call로 부른다.
  opened() { this.readyState = 1; this.onopen?.call(this as unknown as WebSocket, new Event('open')) }
  message(data: unknown) { this.onmessage?.call(this as unknown as WebSocket, new MessageEvent('message', { data: JSON.stringify(data) })) }
  closed(code: number) { this.readyState = 3; this.onclose?.call(this as unknown as WebSocket, new CloseEvent('close', { code })) }
  /** Everything sent, JSON messages parsed and audio frames as 'audio'. */
  // 보낸 것 목록: 글 메시지는 JSON으로 풀고, 소리(바이너리)는 내용 대신 'audio'로 바꿔 순서만 비교하기 쉽게
  // 한다.
  messages(): unknown[] { return this.send.mock.calls.map(([data]) => typeof data === 'string' ? JSON.parse(data) : 'audio') }
}

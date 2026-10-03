import { vi } from 'vitest'
import type { LiveSocket } from '../live/socket'

/** A WebSocket the test drives: open it, deliver server messages, close it with a code. */
export class FakeSocket implements LiveSocket {
  readyState: WebSocket['readyState'] = 0; bufferedAmount = 0
  onopen: WebSocket['onopen'] = null
  onmessage: WebSocket['onmessage'] = null
  onclose: WebSocket['onclose'] = null
  onerror: WebSocket['onerror'] = null
  send = vi.fn<(data: string | ArrayBufferLike | Blob | ArrayBufferView) => void>()
  close = vi.fn(() => { this.readyState = 3 })
  opened() { this.readyState = 1; this.onopen?.call(this as unknown as WebSocket, new Event('open')) }
  message(data: unknown) { this.onmessage?.call(this as unknown as WebSocket, new MessageEvent('message', { data: JSON.stringify(data) })) }
  closed(code: number) { this.readyState = 3; this.onclose?.call(this as unknown as WebSocket, new CloseEvent('close', { code })) }
  /** Everything sent, JSON messages parsed and audio frames as 'audio'. */
  messages(): unknown[] { return this.send.mock.calls.map(([data]) => typeof data === 'string' ? JSON.parse(data) : 'audio') }
}

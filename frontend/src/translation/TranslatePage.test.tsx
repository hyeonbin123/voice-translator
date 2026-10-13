// [파일 역할] 번역 화면(TranslatePage)과 음성 재생 영역(Playback) 테스트 묶음. 실제 ApiClient에 가짜 fetch를
// 붙여 인증 머리글, 요청 본문, 오류 문구 선택, 업로드 형식, 포커스, 취소, blob URL 정리까지 화면 쪽 계약을
// 확인한다.
// [가짜 객체] fetch, speechSynthesis(브라우저 음성), URL.createObjectURL·revokeObjectURL, 미디어 pause를
// 가짜로 바꾼다. beforeEach에서 refresh 토큰으로 세션을 실제로 복원해 access 토큰을 얻은 뒤 fetch 기록을
// 지운다.

import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { ApiClient, REFRESH_KEY } from '../api/client'
import { TranslationApi, MAX_AUDIO_BYTES, type TranslationResult } from './api'
import TranslatePage from './TranslatePage'
import Playback from './Playback'

const result: TranslationResult = {
  id: '080b161c-53d6-4460-992b-f778a5e348cd', mode: 'text', source_lang: 'ko', target_lang: 'en',
  source_text: '안녕하세요', translated_text: 'Hello', stt_model: null, mt_model: 'example-mt',
  tts_model: null, stt_ms: null, mt_ms: 120, tts_ms: null, audio_id: null,
  created_at: '2026-09-13T00:00:00Z', tts_error: 'Speech synthesis is not available',
}
const response = (body: unknown, status = 201) => new Response(JSON.stringify(body), { status })
const fetchMock = vi.fn<typeof fetch>()
let api: TranslationApi
const speak = vi.fn()
const cancel = vi.fn()
class Utterance {
  text: string
  lang = ''
  onend: (() => void) | null = null
  onerror: ((event: { error: string }) => void) | null = null
  constructor(text: string) { this.text = text }
}

beforeEach(async () => {
  sessionStorage.clear()
  sessionStorage.setItem(REFRESH_KEY, 'refresh')
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
  speak.mockReset(); cancel.mockReset()
  vi.stubGlobal('speechSynthesis', { speak, cancel })
  vi.stubGlobal('SpeechSynthesisUtterance', Utterance)
  vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => undefined)
  Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: vi.fn(() => 'blob:translation') })
  Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: vi.fn() })
  fetchMock.mockResolvedValueOnce(response({ access_token: 'access', refresh_token: 'refresh' }, 200))
    .mockResolvedValueOnce(response({ id: 'user', email: 'person@example.com' }, 200))
  const client = new ApiClient()
  await client.initialize()
  api = new TranslationApi(client)
  fetchMock.mockClear()
})
afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.restoreAllMocks() })

// 글 상자에 값을 넣고 '번역하기'를 누르는 도우미.
async function submitText(text = '안녕하세요') {
  const user = userEvent.setup()
  fireEvent.change(screen.getByLabelText(/번역할 글/), { target: { value: text } })
  await user.click(screen.getByRole('button', { name: '번역하기' }))
  return user
}

// 확인: 앞뒤 공백을 자른 글과 방향이 JSON으로 Bearer 토큰과 함께 가고, 결과 제목으로 포커스가 옮겨지며,
// 번역문에 lang이 붙고 처리 시간(120 ms, 실행 안 함 2개)이 보인다. 서버 음성이 없을 때 자동으로 읽지 않고
// 버튼을 눌러야 읽으며 중지할 수 있다. 화면을 닫아도 로그인 토큰은 남는다.
it('submits trimmed Unicode text with authentication, shows the result and timings, and reads it on demand', async () => {
  fetchMock.mockResolvedValueOnce(response(result))
  const view = render(<TranslatePage api={api} />)
  const user = await submitText('  안녕하세요  ')
  expect(await screen.findByRole('heading', { name: '번역 결과' })).toHaveFocus()
  expect(screen.getByText('Hello')).toHaveAttribute('lang', 'en')
  expect(screen.getByText('120 ms')).toBeInTheDocument()
  expect(screen.getAllByText('실행 안 함')).toHaveLength(2)
  const [path, init] = fetchMock.mock.calls[0]
  expect(path).toBe('/api/translate/text')
  expect(new Headers(init?.headers).get('Authorization')).toBe('Bearer access')
  expect(JSON.parse(init?.body as string)).toEqual({ text: '안녕하세요', source_lang: 'ko', target_lang: 'en' })
  expect(speak).not.toHaveBeenCalled()
  await user.click(screen.getByRole('button', { name: '번역문 읽기' }))
  expect(speak.mock.calls[0][0]).toMatchObject({ text: 'Hello', lang: 'en-US' })
  await user.click(screen.getByRole('button', { name: '읽기 중지' }))
  expect(cancel).toHaveBeenCalled()
  view.unmount()
  expect(sessionStorage.length).toBe(1)
})

// 확인: 방향을 영어 → 한국어로 바꾸면 요청 방향이 바뀌고 한국어 음성(ko-KR)으로 읽는다. 이모지 500개는
// UTF-16으로는 1000단위지만 500자로 세어 통과한다(서버와 같은 코드 포인트 단위).
it('changes direction and counts supplementary Unicode characters as one character each', async () => {
  fetchMock.mockResolvedValueOnce(response({ ...result, source_lang: 'en', target_lang: 'ko', translated_text: '안녕하세요' }))
  render(<TranslatePage api={api} />)
  fireEvent.change(screen.getByLabelText('말하거나 입력할 언어'), { target: { value: 'en' } })
  await submitText('😀'.repeat(500))
  expect(JSON.parse(fetchMock.mock.calls[0][1]?.body as string)).toMatchObject({ source_lang: 'en', target_lang: 'ko' })
  await userEvent.click(screen.getByRole('button', { name: '번역문 읽기' }))
  expect(speak.mock.calls[0][0].lang).toBe('ko-KR')
})

// 확인: 공백만 있는 글과 501자는 요청 없이 1~500자 안내를 보인다.
it.each(['   ', '가'.repeat(501)])('rejects invalid text without a request (%#. boundary)', async (text) => {
  render(<TranslatePage api={api} />)
  await submitText(text)
  expect(screen.getByRole('alert')).toHaveTextContent('1~500자')
  expect(fetchMock).not.toHaveBeenCalled()
})

// 확인: 오류 문구는 계약에 있는 '상태 코드:detail' 짝일 때만 그 안내를 고르고(413·422·503의 고정 문자열),
// 짝이 맞지 않거나 계약 밖이면 일반 문구로 떨어진다. 응답에 담긴 입력값('secret-password')이나 다른 경로의
// 문구는 화면에 나오지 않고, 결과 없이 버튼이 다시 켜진다.
// 왜: 서버 문자열을 그대로 보이면 내부 내용이나 입력한 비밀이 화면에 새어 나올 수 있다(api/client.ts의
// checked).
it.each([
  [413, 'Audio file is larger than 10 MB', '10MB 이하'],
  [422, 'Audio could not be decoded', '파일을 읽을 수 없습니다'],
  [422, 'Audio is longer than 30 seconds', '30초 이하'],
  [422, 'No speech was recognized', '말소리를 찾지 못했습니다'],
  [503, 'Translation service is unavailable', '번역 서비스를 사용할 수 없습니다'],
  [422, [{ input: 'secret-password' }], '입력 내용과 번역 방향'],
  [409, 'Email already registered', '요청을 처리하지 못했습니다'],
  [503, 'No speech was recognized', '요청을 처리하지 못했습니다'],
  [500, 'secret-password', '요청을 처리하지 못했습니다'],
] as const)('uses only the contract status/detail pair for %s / %j', async (status, detail, message) => {
  fetchMock.mockResolvedValueOnce(response({ detail }, status))
  render(<TranslatePage api={api} />)
  await submitText()
  expect(await screen.findByRole('alert')).toHaveTextContent(message)
  expect(screen.queryByText(/secret-password|이미 가입된 이메일/)).not.toBeInTheDocument()
  expect(screen.queryByRole('heading', { name: '번역 결과' })).not.toBeInTheDocument()
  expect(screen.getByRole('button', { name: '번역하기' })).toBeEnabled()
})

// 확인: 음성 파일은 multipart로 올라가며 Content-Type을 직접 넣지 않는다(브라우저가 boundary를 붙여야
// 하므로). 파일 이름·형식·내용과 방향 필드가 그대로 가고, 응답의 인식 시간(200 ms)이 보인다.
it('uploads audio as multipart without setting its boundary and displays speech timing', async () => {
  fetchMock.mockResolvedValueOnce(response({ ...result, mode: 'speech', stt_model: 'example-stt', stt_ms: 200 }))
  render(<TranslatePage api={api} />)
  const user = userEvent.setup()
  await user.click(screen.getByLabelText('음성 입력'))
  const file = new File(['audio'], 'voice.wav', { type: 'audio/wav' })
  await user.upload(screen.getByLabelText('음성 파일', { exact: true }), file)
  await user.click(screen.getByRole('button', { name: '번역하기' }))
  expect(await screen.findByText('200 ms')).toBeInTheDocument()
  const [path, init] = fetchMock.mock.calls[0]
  expect(path).toBe('/api/translate/speech')
  expect(new Headers(init?.headers).has('Content-Type')).toBe(false)
  expect(new Headers(init?.headers).get('Authorization')).toBe('Bearer access')
  const form = init?.body as FormData
  const uploaded = form.get('audio') as File
  expect(uploaded).toMatchObject({ name: 'voice.wav', type: 'audio/wav', size: file.size })
  const contents = await new Promise((resolve) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result)
    reader.readAsText(uploaded)
  })
  expect(contents).toBe('audio')
  expect(form.get('source_lang')).toBe('ko')
  expect(form.get('target_lang')).toBe('en')
})

// 확인: 두 사람 대화를 고르면 방향 선택과 '번역하기' 버튼이 사라지고 두 사람 대화 패널이 나온다.
it('offers two-person dialog without a manual language direction', async () => {
  render(<TranslatePage api={api} />)
  await userEvent.click(screen.getByLabelText('두 사람 대화'))
  expect(screen.getByRole('heading', { name: '두 사람 대화' })).toBeInTheDocument()
  expect(screen.queryByLabelText('말하거나 입력할 언어')).not.toBeInTheDocument()
  expect(screen.queryByRole('button', { name: '번역하기' })).not.toBeInTheDocument()
})

// 확인: 숨긴 파일 입력 대신 버튼으로 고르고, 고른 파일 이름이 상태 문구와 버튼 설명으로 읽힌다. 입력값은 비워
// 같은 파일을 다시 고를 수 있다. 파일 창을 취소해도 고른 파일은 유지되고, 방향을 바꾸면 지워진다(다른 언어로
// 녹음한 파일을 그대로 보내지 않게).
it('announces the selected file, preserves it on cancel, and resets it when direction changes', async () => {
  render(<TranslatePage api={api} />)
  const user = userEvent.setup()
  await user.click(screen.getByLabelText('음성 입력'))
  const input = screen.getByLabelText('음성 파일', { exact: true })
  const button = screen.getByRole('button', { name: '또는 음성 파일 선택' })
  const status = screen.getByRole('status')
  expect(input).not.toBeVisible()
  expect(status).toHaveTextContent('선택한 음성이 없습니다.')
  expect(status).toHaveAttribute('aria-atomic', 'true')
  const file = new File(['audio'], 'voice.wav', { type: 'audio/wav' })
  await user.upload(input, file)
  expect(status).toHaveTextContent('선택한 파일: voice.wav')
  expect(button).toHaveAccessibleDescription(/선택한 파일: voice.wav/)
  expect(input).toHaveValue('')
  fireEvent(input, new Event('cancel'))
  fireEvent.change(input, { target: { files: [] } })
  expect(status).toHaveTextContent('선택한 파일: voice.wav')
  expect(screen.getByRole('button', { name: '번역하기' })).toBeEnabled()
  await user.selectOptions(screen.getByLabelText('말하거나 입력할 언어'), 'en')
  expect(status).toHaveTextContent('선택한 음성이 없습니다.')
  expect(screen.getByRole('button', { name: '번역하기' })).toBeDisabled()
  await user.upload(input, file)
  expect(status).toHaveTextContent('선택한 파일: voice.wav')
  expect(screen.getByRole('button', { name: '번역하기' })).toBeEnabled()
  expect(fetchMock).not.toHaveBeenCalled()
})

// 확인: 빈 파일이나 크기 한도를 넘는 파일을 고르면 오류를 보이고, 앞서 고른 파일도 지워 '번역하기'를 끈다.
// 요청은 보내지 않는다.
it.each([0, MAX_AUDIO_BYTES + 1])('rejects an empty or oversized upload: %s bytes', async (size) => {
  render(<TranslatePage api={api} />)
  await userEvent.click(screen.getByLabelText('음성 입력'))
  await userEvent.upload(screen.getByLabelText('음성 파일', { exact: true }), new File(['valid'], 'previous.wav', { type: 'audio/wav' }))
  const file = new File(['x'], 'voice.wav', { type: 'audio/wav' })
  Object.defineProperty(file, 'size', { value: size })
  await userEvent.upload(screen.getByLabelText('음성 파일', { exact: true }), file)
  expect(screen.getByRole('alert')).toHaveTextContent('비어 있지 않은 10MB 이하')
  expect(screen.getByRole('button', { name: '번역하기' })).toBeDisabled()
  expect(screen.getByRole('status')).toHaveTextContent('선택한 음성이 없습니다.')
  expect(screen.queryByText(/previous.wav/)).not.toBeInTheDocument()
  expect(fetchMock).not.toHaveBeenCalled()
})

// 확인: 번역 중에는 버튼과 방향 선택이 잠겨 요청이 하나만 가고, 화면을 떠나면 그 요청이 취소되며 늦게 온
// 응답은 반영하지 않는다.
it('locks duplicate submissions and aborts an in-flight translation when leaving', async () => {
  let resolve!: (value: Response) => void
  fetchMock.mockReturnValueOnce(new Promise<Response>((done) => { resolve = done }))
  const view = render(<TranslatePage api={api} />)
  await submitText()
  expect(screen.getByRole('button', { name: '번역 중…' })).toBeDisabled()
  expect(screen.getByLabelText('말하거나 입력할 언어')).toBeDisabled()
  expect(fetchMock).toHaveBeenCalledTimes(1)
  const signal = fetchMock.mock.calls[0][1]?.signal
  view.unmount()
  expect(signal?.aborted).toBe(true)
  await act(async () => { resolve(response(result)) })
  expect(screen.queryByText('Hello')).not.toBeInTheDocument()
})

// Playback 확인: 서버 음성을 Bearer 토큰으로 받아 blob URL로 재생 컨트롤에 넣고, 서버 음성이 있으면 브라우저
// 읽기 버튼은 없다. 사라질 때 재생을 멈추고 URL을 해제한다.
it('fetches server audio with a bearer token and releases its blob URL on removal', async () => {
  const blob = new Blob(['wav'], { type: 'audio/wav' })
  fetchMock.mockResolvedValueOnce(new Response(blob))
  const view = render(<Playback result={{ ...result, audio_id: 'a17a7bf4-ce62-4d8b-9309-a220b1868299' }} api={api} />)
  await waitFor(() => expect(screen.getByLabelText('번역 음성')).toHaveAttribute('src', 'blob:translation'))
  expect(fetchMock.mock.calls[0][0]).toBe('/api/audio/a17a7bf4-ce62-4d8b-9309-a220b1868299')
  expect(new Headers(fetchMock.mock.calls[0][1]?.headers).get('Authorization')).toBe('Bearer access')
  expect(screen.queryByRole('button', { name: '번역문 읽기' })).not.toBeInTheDocument()
  view.unmount()
  expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:translation')
  expect(HTMLMediaElement.prototype.pause).toHaveBeenCalled()
})

// Playback 확인: 음성이 404면 계약 안내를 보이고, '음성 다시 불러오기'를 누르면 새로 받아 재생할 수 있다.
it('handles missing server audio and retries with a fresh fetch', async () => {
  fetchMock.mockResolvedValueOnce(response({ detail: 'Audio not found' }, 404))
    .mockResolvedValueOnce(new Response(new Blob(['wav'])))
  render(<Playback result={{ ...result, audio_id: 'audio-id' }} api={api} />)
  expect(await screen.findByRole('alert')).toHaveTextContent('없거나 삭제')
  await userEvent.click(screen.getByRole('button', { name: '음성 다시 불러오기' }))
  await waitFor(() => expect(screen.getByLabelText('번역 음성')).toHaveAttribute('src', 'blob:translation'))
  expect(fetchMock).toHaveBeenCalledTimes(2)
})

// Playback 확인: 음성을 받는 중에 사라지면 요청을 취소하고, 늦게 도착해도 blob URL을 만들지 않는다(해제되지
// 않는 URL이 남지 않게).
it('does not create a blob URL if the audio fetch resolves after unmount', async () => {
  let resolve!: (value: Response) => void
  fetchMock.mockReturnValueOnce(new Promise<Response>((done) => { resolve = done }))
  const view = render(<Playback result={{ ...result, audio_id: 'audio-id' }} api={api} />)
  view.unmount()
  expect(fetchMock.mock.calls[0][1]?.signal?.aborted).toBe(true)
  await act(async () => { resolve(new Response(new Blob(['wav']))) })
  expect(URL.createObjectURL).not.toHaveBeenCalled()
})

// Playback 확인: 합성만 실패한 결과는 그 사실을 알리고, 브라우저에 음성 읽기 기능도 없으면 읽기 버튼을 눌렀을
// 때 안내한다. 서버 요청은 하지 않는다.
it('explains a TTS-only failure and handles unavailable browser voices', async () => {
  vi.stubGlobal('speechSynthesis', undefined)
  render(<Playback result={{ ...result, tts_error: 'Speech synthesis failed' }} api={api} />)
  expect(screen.getByText(/서버 음성 합성에 실패/)).toBeInTheDocument()
  await userEvent.click(screen.getByRole('button', { name: '번역문 읽기' }))
  expect(screen.getByRole('alert')).toHaveTextContent('음성 읽기를 사용할 수 없습니다')
  expect(fetchMock).not.toHaveBeenCalled()
})

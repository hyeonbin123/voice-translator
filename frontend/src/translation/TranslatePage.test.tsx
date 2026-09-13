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

async function submitText(text = '안녕하세요') {
  const user = userEvent.setup()
  fireEvent.change(screen.getByLabelText(/번역할 글/), { target: { value: text } })
  await user.click(screen.getByRole('button', { name: '번역하기' }))
  return user
}

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

it('changes direction and counts supplementary Unicode characters as one character each', async () => {
  fetchMock.mockResolvedValueOnce(response({ ...result, source_lang: 'en', target_lang: 'ko', translated_text: '안녕하세요' }))
  render(<TranslatePage api={api} />)
  fireEvent.change(screen.getByLabelText('말하거나 입력할 언어'), { target: { value: 'en' } })
  await submitText('😀'.repeat(500))
  expect(JSON.parse(fetchMock.mock.calls[0][1]?.body as string)).toMatchObject({ source_lang: 'en', target_lang: 'ko' })
  await userEvent.click(screen.getByRole('button', { name: '번역문 읽기' }))
  expect(speak.mock.calls[0][0].lang).toBe('ko-KR')
})

it.each(['   ', '가'.repeat(501)])('rejects invalid text without a request (%#. boundary)', async (text) => {
  render(<TranslatePage api={api} />)
  await submitText(text)
  expect(screen.getByRole('alert')).toHaveTextContent('1~500자')
  expect(fetchMock).not.toHaveBeenCalled()
})

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

it('handles missing server audio and retries with a fresh fetch', async () => {
  fetchMock.mockResolvedValueOnce(response({ detail: 'Audio not found' }, 404))
    .mockResolvedValueOnce(new Response(new Blob(['wav'])))
  render(<Playback result={{ ...result, audio_id: 'audio-id' }} api={api} />)
  expect(await screen.findByRole('alert')).toHaveTextContent('없거나 삭제')
  await userEvent.click(screen.getByRole('button', { name: '음성 다시 불러오기' }))
  await waitFor(() => expect(screen.getByLabelText('번역 음성')).toHaveAttribute('src', 'blob:translation'))
  expect(fetchMock).toHaveBeenCalledTimes(2)
})

it('does not create a blob URL if the audio fetch resolves after unmount', async () => {
  let resolve!: (value: Response) => void
  fetchMock.mockReturnValueOnce(new Promise<Response>((done) => { resolve = done }))
  const view = render(<Playback result={{ ...result, audio_id: 'audio-id' }} api={api} />)
  view.unmount()
  expect(fetchMock.mock.calls[0][1]?.signal?.aborted).toBe(true)
  await act(async () => { resolve(new Response(new Blob(['wav']))) })
  expect(URL.createObjectURL).not.toHaveBeenCalled()
})

it('explains a TTS-only failure and handles unavailable browser voices', async () => {
  vi.stubGlobal('speechSynthesis', undefined)
  render(<Playback result={{ ...result, tts_error: 'Speech synthesis failed' }} api={api} />)
  expect(screen.getByText(/서버 음성 합성에 실패/)).toBeInTheDocument()
  await userEvent.click(screen.getByRole('button', { name: '번역문 읽기' }))
  expect(screen.getByRole('alert')).toHaveTextContent('음성 읽기를 사용할 수 없습니다')
  expect(fetchMock).not.toHaveBeenCalled()
})

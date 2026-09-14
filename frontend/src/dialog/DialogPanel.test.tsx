import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { expect, it, vi } from 'vitest'
import { TranslationApi } from '../translation/api'
import DialogPanel from './DialogPanel'

it('switches between bubbles and face-to-face layout and swaps the viewer languages', async () => {
  const api = new TranslationApi({ request: vi.fn() })
  render(<DialogPanel api={api} onActiveChange={vi.fn()} />)
  expect(screen.getByText(/한국어는 왼쪽, 영어는 오른쪽/)).toBeInTheDocument()
  const user = userEvent.setup()
  await user.click(screen.getByRole('button', { name: '마주 보기' }))
  const layout = screen.getByLabelText('마주 보기 대화')
  expect(layout).toHaveClass('face-layout')
  expect(screen.getByRole('region', { name: '한국어 화자 쪽 · 맞은편' })).toHaveClass('face-pane-top')
  expect(screen.getByRole('region', { name: '영어 화자 쪽 · 가까운 쪽' })).not.toHaveClass('face-pane-top')
  await user.click(screen.getByRole('button', { name: '칸 언어 바꾸기' }))
  expect(screen.getByRole('region', { name: '영어 화자 쪽 · 맞은편' })).toHaveClass('face-pane-top')
  expect(screen.getByRole('region', { name: '한국어 화자 쪽 · 가까운 쪽' })).not.toHaveClass('face-pane-top')
  await user.click(screen.getByRole('button', { name: '말풍선 보기' }))
  expect(screen.getByLabelText('두 사람 대화 내용')).toBeInTheDocument()
})

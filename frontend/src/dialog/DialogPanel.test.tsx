// [파일 역할] 두 사람 대화 화면(DialogPanel)의 보기 전환 테스트. 마이크를 열지 않고(시작 버튼을 누르지 않음)
// 화면 배치만 확인한다. 흐름은 session.test.ts가 따로 검사한다.

import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { expect, it, vi } from 'vitest'
import { TranslationApi } from '../translation/api'
import DialogPanel from './DialogPanel'

// 확인: 처음에는 말풍선 안내가 보이고, '마주 보기'를 누르면 맞은편(한국어, 돌려진 칸)과 가까운 쪽(영어) 칸이
// 생기며, '칸 언어 바꾸기'로 두 칸의 언어가 서로 바뀌고, '말풍선 보기'로 목록으로 돌아온다.
// 왜: 휴대폰을 두 사람 사이에 놓는 방향에 따라 어느 칸이 뒤집혀야 하는지가 달라지므로, 칸 이름(aria-label)과
// 뒤집힘 클래스가 함께 바뀌는지 본다.
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

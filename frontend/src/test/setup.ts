// [파일 역할] Vitest가 모든 테스트 파일보다 먼저 실행하는 준비 파일(vite.config.ts의 test.setupFiles).
// [하는 일] (1) toBeInTheDocument, toHaveAttribute 같은 DOM 비교 함수(jest-dom)를 Vitest의 expect에 등록한다.
// (2) 테스트가 끝날 때마다 그려 둔 React 화면을 치워, 앞 테스트의 화면·이벤트 처리기가 다음 테스트에 남지
// 않게 한다.

import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach } from 'vitest'

// Vitest globals are off, so Testing Library can't register its own cleanup.
// cleanup은 render로 붙인 컴포넌트를 모두 떼어 낸다(unmount). 떼어 낼 때 각 컴포넌트의 effect 정리 함수가
// 돌아 마이크·소켓 같은 가짜 자원도 닫힌다.
afterEach(() => {
  cleanup()
})

// [파일 역할] 로그인 상태(Session)를 React 화면에서 읽는 훅.
// [왜 이 방식인가] 로그인 상태는 React 바깥의 ApiClient 객체(api/client.ts)가 들고 있고, 토큰 갱신 실패나
// 로그아웃처럼 화면과 무관한 곳에서도 바뀐다. useSyncExternalStore는 이런 바깥 저장소를 구독해, 상태가 바뀌면
// 그 상태를 쓰는 모든 화면을 같은 값으로 다시 그리게 하는 React 표준 훅이다. 별도 상태 관리 라이브러리나
// Context 없이 해결된다.

import { useSyncExternalStore } from 'react'
import type { ApiClient } from '../api/client'

// 입력: ApiClient. 출력: 현재 Session(status: loading·anonymous·authenticated, user, expired).
// client.subscribe로 변경 알림을 받고 client.getSnapshot으로 현재 값을 읽는다. 두 함수는 ApiClient에서 화살표
// 함수 필드로 만들어져 this가 고정되고 매 렌더마다 같은 함수라, 구독이 렌더마다 다시 걸리지 않는다.
// 호출 위치: App.tsx(ProtectedLayout, AppRoutes), auth/AuthPage.tsx.
export function useSession(client: ApiClient) {
  return useSyncExternalStore(client.subscribe, client.getSnapshot)
}

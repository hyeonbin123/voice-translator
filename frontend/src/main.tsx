// [파일 역할] 화면의 진입점. index.html의 script가 이 파일을 불러오고, 여기서 React 앱을 #root 요소에 붙인다.
// [연결] 전체 스타일(styles.css)을 한 번 불러오고, 라우팅과 화면 구성은 App.tsx에 맡긴다.

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'
import App from './App.tsx'

// React 18 이후의 createRoot로 #root에 앱을 그린다. 끝의 !는 index.html에 #root가 반드시 있으므로 null이
// 아니라고 TypeScript에 알려 주는 것이다.
// StrictMode는 개발 중에만 effect를 한 번 더 실행했다 정리해 보는 등 추가 검사를 한다. 이 앱은
// 마이크·웹소켓·오디오 재생을 effect에서 열고 닫으므로, 정리 함수가 빠진 실수를 개발 중에 드러내 준다. 배포
// 빌드의 동작에는 영향이 없다.
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)

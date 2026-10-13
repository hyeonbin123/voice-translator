// [파일 역할] 프런트엔드 린트 규칙 설정(flat config 형식)이다. `npm run lint`가 이 파일을 읽어 src의
// TypeScript·TSX 코드를 검사하고, GitHub Actions CI도 같은 명령을 돌린다(.github/workflows/ci.yml).
// [주요 구성] 빌드 결과물(dist) 제외 → .ts/.tsx 파일에 자바스크립트 기본 권장 규칙, TypeScript 권장 규칙,
// React 훅 규칙, Vite 빠른 새로 고침 규칙을 함께 적용 → 브라우저 전역 변수를 알려 준다.
// [연결] 프로젝트 뼈대를 만들 때(f0900f3) 넣은 뒤 바뀌지 않았고, Vite의 React + TypeScript 템플릿과 같은
// 구성이다.

// 규칙 묶음 가져오기: js는 자바스크립트 기본 권장 규칙, globals는 브라우저 전역 이름 목록,
// reactHooks·reactRefresh는 React 전용 규칙, 다섯째 줄은 TypeScript 파서와 권장 규칙, 마지막 줄은 설정을
// 배열로 묶는 도우미(defineConfig)와 검사 제외 목록 도우미(globalIgnores)다.
import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'
import { defineConfig, globalIgnores } from 'eslint/config'

// 설정 객체 배열을 내보낸다. 배열의 앞에서부터 차례로 겹쳐 적용된다.
export default defineConfig([
  // vite build가 만든 dist 폴더는 압축된 결과물이라 검사하지 않는다.
  globalIgnores(['dist']),
  {
    // 아래 규칙은 확장자가 .ts와 .tsx인 소스에만 적용한다. 이 저장소의 화면 코드는 거의 모두 TypeScript다.
    files: ['**/*.{ts,tsx}'],
    // - js.configs.recommended: 쓰지 않는 변수, 도달할 수 없는 코드 같은 기본 실수를 잡는다
    // - 둘째 줄: TypeScript 문법을 이해하는 파서와 권장 규칙(any 남용, 쓰지 않는 타입 등)
    // - reactHooks: 훅을 조건문 안에서 부르지 않는지, useEffect 의존성 배열이 빠지지 않았는지 검사한다. 화면
    //   대부분이 useEffect·useRef로 마이크·소켓 수명을 다루므로 중요하다
    // - reactRefresh.configs.vite: 컴포넌트 파일이 컴포넌트만 내보내는지 검사해, Vite 개발 서버의 빠른 새로
    //   고침(상태를 유지한 채 컴포넌트만 바꾸기)이 깨지지 않게 한다
    extends: [
      js.configs.recommended,
      tseslint.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    // window, document, navigator 같은 브라우저 전역을 선언 없이 써도 '정의되지 않은 이름' 오류로 보지 않게
    // 한다.
    languageOptions: {
      globals: globals.browser,
    },
  },
])

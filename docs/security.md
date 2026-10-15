# 보안 점검 기록

이 서비스는 한 PC에서 돌리는 것을 기본으로 한다(DB는 127.0.0.1에만 열리고, 화면·API는 `WEB_HOST_PORT`의 HTTP). 이 문서는 무엇을 확인했고 무엇을 받아들인 위험으로 남겼는지 적는다. 날짜가 붙은 항목은 그날의 결과이며 이후 바뀐 것은 덧붙여 적는다.

## 재현성 고정 (2026-10-15)

같은 파일로 다시 만들었을 때 같은 서버가 뜨도록, 받아 오는 것들을 판 번호나 해시로 고정했다.

| 대상 | 고정 방법 | 확인 |
|---|---|---|
| 음성 인식 모델 (faster-whisper large-v3-turbo 변환본) | Hugging Face 저장소 커밋 `0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf` (`backend/app/services/stt.py`의 `TURBO_REVISION`, 측정 도구도 같은 값) | 측정에 쓴 PC 캐시와 Docker `hfcache` 볼륨의 스냅숏이 같은 커밋이고, `model.bin`의 SHA-256이 허브의 파일 해시와 같다(`e76620f8…`). 저장소 이름은 그대로 둬 다시 받지 않는다(허브에서는 새 소유자 이름으로 넘어가지만 같은 커밋) |
| Docker 기본 이미지 | 태그와 함께 digest(sha256): `python:3.11-slim-bookworm`, `ghcr.io/astral-sh/uv:0.12`(`backend/Dockerfile`), `node:24-alpine`, `nginx:stable-alpine`(`frontend/Dockerfile`), `postgres:16`, `ollama/ollama:0.35.1`(`docker-compose.yml`, CI의 postgres도 같은 digest) | python 기본 이미지는 고정할 때 이미 빌드돼 있던 api 이미지의 아래 층 세 개와 같은 이미지다(Python 3.11.17). 보안 수정을 받으려면 태그를 다시 받아 digest를 새로 적는다(`backend/Dockerfile` 머리 주석) |
| 브라우저 말소리 판정 런타임 | `onnxruntime-web` 1.30.0 (`frontend/package.json`, 정확한 판) | 1.29.0에서 올리기 전후로 같은 소리를 Edge와 Playwright Chromium에서 돌려 비교했다: FLEURS validation 영어·한국어 세션(조용함, SNR 10dB 분홍 잡음) 15,834조각의 말소리 확률이 두 판에서 비트까지 같았고, Python onnxruntime 1.30.0과의 차이는 최대 1.2×10⁻⁵, 0.5 기준이 바뀐 조각은 0개였다(관문: 1×10⁻⁴ 이하, 뒤집힘 0). 그래서 대화 모드의 말 끝 판정 측정(7절)은 다시 재지 않는다 |
| Python 의존성 | `backend/uv.lock`(CI와 이미지 모두 `--locked`) | — |
| npm 의존성 | `frontend/package-lock.json`(`npm ci`) | 아래 '의존성 취약점'의 개발 도구 두 개만 lock에서 올렸고, 빌드 결과(`dist/`)가 올리기 전과 바이트까지 같음을 확인했다 |

## 의존성 취약점 (2026-10-15)

`pip-audit` 2.10.1(PyPI 권고 데이터베이스)과 `npm audit`으로 lock 그대로의 의존성을 검사했다. Python은 서버 환경(gpu + tts 그룹)과 측정 도구 환경(gpu + eval + dev 그룹)을 따로 내보내 검사했다(두 그룹은 함께 설치할 수 없다).

**고친 것** (lock에서 판만 올림, 백엔드 테스트 558개 통과)

| 패키지 | 판 | 권고 | 이 서비스에 해당했나 |
|---|---|---|---|
| PyJWT | 2.14.0 → 2.15.1 | GHSA-42vr-xj54-vc7v (JWKS 흐름에서 깊이 중첩된 내용이 정의되지 않은 예외로 새어 나옴), GHSA-x33g-cr3x-6449 (OKP 비밀 JWK의 공개 키 확인 누락) | 아니다. 이 서비스는 JWKS와 OKP 키를 쓰지 않고, HS256 하나로 서명을 먼저 검증한 뒤에야 내용을 읽으므로 비밀 키 없이는 중첩된 내용을 넣을 수 없다. 인증 경로의 패키지라 올렸다 |
| urllib3 | 2.7.0 → 2.8.0 | GHSA-vxq7-64xx-v4gw·GHSA-gh4c-6fx4-qh6g (악의적 서버의 chunked 응답 스트리밍에서 메모리·무한 반복), GHSA-8988-9cw3-xx77 (HTTPS 프록시 TLS 설정 혼동) | 모델을 처음 받을 때(Hugging Face) 쓰는 경로다. 프록시는 쓰지 않는다 |
| Werkzeug | 3.1.8 → 3.1.9 | GHSA-g6x2-hccm-hh4m (Windows 장치 이름으로 `safe_join` 우회) | 아니다(MeloTTS가 끌어오는 tensorboard의 의존성이고 서비스가 쓰지 않음). 고치는 판이 있어 올렸다 |
| brace-expansion (npm, 개발 도구) | 5.0.9 → 5.0.12 | GHSA-q2hr-2g5m-vwhr 등 3건 (중괄호 확장 DoS) | 아니다(eslint 등 개발 도구 안). 배포 파일에 들어가지 않는다 |
| source-map-js (npm, 개발 도구) | 1.2.1 → 1.2.2 | GHSA-68fv-2mgg-jv7q (색인 소스 맵 DoS) | 아니다(빌드 도구 안) |

**받아들인 것** (고치지 않음, 이유)

| 패키지 | 권고 | 이유 |
|---|---|---|
| transformers 4.27.4 (서버 환경) | 권고 30건(설명 문구를 대강 나눔: 신뢰할 수 없는 체크포인트·원격 코드 역직렬화 약 17건, 토크나이저 정규식 ReDoS 약 10건, 그 밖에 임시 파일·이미지 URL 검증·`save_pretrained` 경로 3건) | MeloTTS가 이 판을 고정해 올릴 수 없다. 서버는 고정된 공개 모델(MeloTTS·BERT 토크나이저)만 읽고, 사용자 입력으로 모델 경로·URL을 정하지 않으며, 토크나이저에 들어가는 글은 번역문(번역 모델이 만든 글)이다. 정규식 ReDoS는 번역문 길이가 입력 제한(글자 입력 500자, 음성 30초)에 묶여 영향이 제한된다(재지 않음) |
| torch 2.11.0+cu128 (서버 환경) | PYSEC-2025-194 (`torch.jit.script` 메모리 손상, 로컬) | 서비스는 `torch.jit.script`를 부르지 않는다. 이 판은 MeloTTS와 CUDA 12.8에 맞춘 tts 그룹 고정이다. pip-audit는 PyPI에 없는 `+cu128` 판을 검사하지 못해 같은 2.11.0을 따로 검사했다 |
| nltk 3.10.3 | PYSEC-2026-3740 (모델 저장·읽기 API의 경로 제한 우회) | 고친 판이 아직 없다. 서비스는 사용자 입력으로 NLTK 모델 경로를 정하지 않는다(이미지에 미리 넣은 데이터만 읽음) |
| setuptools 81.0.0 | PYSEC-2026-3447 (macOS에서 sdist를 만들 때 MANIFEST.in 제외 규칙 우회) | 이 저장소는 sdist를 만들지 않는다. 런타임 경로가 아니다 |

CI의 `audit` 작업이 같은 검사를 push마다 돌린다. 위 '받아들인 것'만 제외하고, 새 권고가 나오면 그 작업이 실패로 표시된다. 처음에는 보고만 하므로(워크플로 전체는 통과) 실패를 보면 이 표에 판단을 더하고 고치거나 받아들인다.

## 화면 서버(nginx)의 보안 머리글과 ZAP 기준 스캔 (2026-10-15)

`frontend/nginx.conf`에 보안 머리글·압축·캐시를 넣고, OWASP ZAP 기준 스캔(baseline: 수동 검사, 공격 요청 없음)을 바꾸기 전후로 한 번씩 돌렸다. 이 저장소에서 처음 한 동적 검사다.

**머리글** (모든 응답, 오류 포함. location이 캐시 머리글을 더해도 `add_header_inherit merge`로 함께 나간다 — nginx 1.29.3부터 있는 지시어, 배포 이미지는 1.30.6)

| 머리글 | 값 | 이유 |
|---|---|---|
| `Content-Security-Policy` | `default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self'; img-src 'self'; media-src 'self' blob:; connect-src 'self'; worker-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'` | 화면의 스크립트·스타일·WASM·모델·워클릿은 모두 같은 출처의 빌드 파일이고, 실행 중에 외부로 나가지 않는다. `'wasm-unsafe-eval'`은 onnxruntime-web이 WebAssembly를 컴파일하는 데만 쓰인다(일반 eval은 막힌 채). `media-src blob:`은 번역 음성을 인증 요청으로 받아 blob URL로 재생하기 때문이다. 인라인 스크립트·스타일과 `data:` 스크립트를 허용하지 않아, XSS로 끼워 넣은 스크립트가 실행되기 어렵다 |
| `X-Frame-Options`, CSP `frame-ancestors 'none'` | `DENY` | 다른 사이트의 iframe에 넣지 못하게(클릭재킹) |
| `Permissions-Policy` | `microphone=(self)`, 그 밖의 장치·결제 기능 끔 | 마이크는 이 화면만 쓴다 |
| `X-Content-Type-Options` | `nosniff` | MIME 추측 실행 막기 |
| `Referrer-Policy` | `no-referrer` | 다른 곳으로 주소를 보내지 않음 |
| `Cross-Origin-Opener-Policy` / `-Resource-Policy` | `same-origin` | 다른 출처 창과 맥락을 나누고, 이 서버의 파일을 다른 출처 페이지가 끌어다 쓰지 못하게 |
| `Server` | `nginx`(판 번호 없음, `server_tokens off`) | 판 번호로 알려진 취약점을 고르기 어렵게 |

- **CSP에 맞춘 화면 수정**: 마이크 워클릿(`capture.worklet.js`, 2.5KB)을 Vite가 4KB 미만이라 `data:` URL로 묶어 넣고 있었다. CSP(`script-src 'self'`)는 `data:` 스크립트를 막으므로, 그대로였다면 배포 화면에서 대화 모드·동시통역·두 사람 대화의 마이크가 열리지 않았다. `?url&no-inline`으로 가져와 `/assets/capture.worklet-<해시>.js` 파일로 내보낸다(`frontend/src/conversation/microphone.ts`). 같은 CSP 아래에서 `data:` 워클릿을 올리면 막히는 것도 대조로 확인했다
- **압축과 캐시**: gzip(5단계)을 켰다. 대화 모드가 처음 받는 onnxruntime-web WASM 14.2MB가 3.72MB로 전송된다(재어 봄). Silero 모델(.onnx)은 86%까지만 줄어 압축하지 않는다. 이름에 내용 해시가 붙은 `/assets/` 파일은 1년 `immutable`로 캐시하고, 이름이 고정된 라이선스 원문 두 개와 `index.html`은 `no-cache`(매번 확인)다
- HSTS는 넣지 않았다: 기본 배포는 같은 PC의 HTTP다. HTTPS 앞단을 두는 쪽에서 정한다
- `Cross-Origin-Embedder-Policy`는 넣지 않았다: 브라우저 추론이 스레드 하나(`env.wasm.numThreads = 1`)라 교차 출처 격리가 필요 없고, 다른 출처 자원도 쓰지 않는다. 넣으면 앞으로 다른 출처 자원을 하나라도 쓸 때 조용히 막힌다

**확인 방법과 결과**
- 단위 테스트(`frontend/src/test/nginx.test.ts`): 머리글 값, merge, gzip 대상, 캐시, 워클릿 import를 글자로 확인한다(CI)
- 실제 브라우저(Microsoft Edge, 가짜 마이크 장치): 배포 nginx 설정·이미지로 띄운 화면에서 시험 계정으로 로그인 → 대화 모드 시작 → 동시통역 시작 → 글자 번역 → 기록 화면을 거치는 동안 CSP 위반 0건. 같은 페이지에서 배포된 Silero 런타임 청크를 불러 실제 WASM 세션으로 한 조각을 추론하고(확률 0.024), 워클릿 파일을 올리는 것도 성공했다. API는 모델 없이(`LOAD_MODELS=false`) 띄워 번역 요청은 503이다(검사 대상이 아님)
- curl: 화면·자산·API 응답 모두 위 머리글이 붙고, WASM은 `Content-Encoding: gzip`, 해시 자산은 `immutable`, 없는 자산은 404

**ZAP 기준 스캔** (OWASP ZAP 2.17.0 `zap-baseline.py`, Docker 이미지 `zaproxy/zap-stable@sha256:7aaa659b…`, 일반 스파이더 + AJAX 스파이더 2분, 대상 `http://web/`. 수동 검사만 하고 공격 요청은 보내지 않는다)

| 위험도 | 바꾸기 전 (2026-10-14 빌드한 이미지) | 바꾼 뒤 |
|---|---|---|
| 높음 | 0 | 0 |
| 중간 | 2: CSP 없음, 클릭재킹 머리글 없음 | 0 |
| 낮음 | 6: COEP·COOP·CORP 없음, Permissions-Policy 없음, Server 판 번호 노출, X-Content-Type-Options 없음 | 1: COEP 없음(위 이유로 넣지 않음) |
| 정보 | 4: 의심스러운 주석(번들 안 라이브러리 주석), 최신 웹 앱(SPA라 AJAX 스파이더를 쓰라는 안내), 저장·캐시 가능한 내용, 저장은 되지만 캐시되지 않는 내용 | 4: 같은 항목 |

- 통과 규칙: 바꾸기 전 58개, 바꾼 뒤 63개(실패 0)
- 범위의 한계: 로그인하지 않은 상태의 수동 검사다. 인증이 필요한 API(번역·기록·음성)와 웹소켓에 대한 능동 검사(공격 요청)는 하지 않았다. API의 개인 데이터 응답은 서버가 이미 `Cache-Control: no-store`(토큰 응답)·`private, no-store`(번역 음성)를 붙이고, 화면은 API를 `cache: 'no-store'`로 부른다

**다시 돌리는 방법** (모델·GPU 없이. 포트 18080은 이 PC에서만)
```bash
# 1. 이미지: docker build -t voice-translator-web frontend  (API는 docker compose build api)
# 2. 모델 없이 DB·API·화면을 따로 띄운다(Compose 프로젝트 이름을 나눠 기존 볼륨을 건드리지 않음)
#    db: postgres:16 이미지, api: LOAD_MODELS=false, DATABASE_URL이 db를 가리키게, web: 127.0.0.1:18080:80
# 3. 같은 Docker 네트워크에서 ZAP 기준 스캔
docker run --rm --network <그 네트워크> -v "$PWD/zap-out:/zap/wrk:rw" \
  zaproxy/zap-stable@sha256:7aaa659b0d43078febd82e29bad112285c370727e86ab8340444220e17d9f0d2 \
  zap-baseline.py -t http://web/ -j -m 2 -I -J report.json -r report.html
```

## 받아들인 위험: refresh 토큰을 sessionStorage에 둠 (2026-10-15 확인, 사용자 결정 대기)

- **지금**: access 토큰(30분)은 화면의 메모리에, refresh 토큰(7일)은 sessionStorage에 둔다(탭을 닫으면 로그아웃, `frontend/src/api/client.ts`). 서버에는 토큰 폐기 목록이 없어(`backend/app/routers/auth.py`) 갱신 전 refresh 토큰도 만료까지 유효하고, 로그아웃은 화면에서 지우는 것뿐이다
- **위험**: 화면에 XSS가 생기면 스크립트가 sessionStorage의 refresh 토큰을 읽어 갈 수 있고, 그 토큰으로 7일 동안 새 access 토큰을 받을 수 있다. 서버 쪽에서 그 토큰 하나만 끊을 방법이 없다(`JWT_SECRET_KEY`를 바꾸면 모든 사용자가 로그아웃된다)
- **이번에 한 것**: 위 CSP로 끼워 넣은 스크립트가 실행되기 어렵게 했다(XSS의 주된 결과를 막는 장치). 화면은 React가 글자를 이스케이프하고 `dangerouslySetInnerHTML`을 쓰지 않는다
- **고치지 않은 이유**: 근본 해결(refresh 토큰을 `HttpOnly; Secure; SameSite=Strict; Path=/api/auth` 쿠키로 옮기고, 서버가 토큰을 회전·폐기하는 표를 두고, 로그아웃 경로를 더함)은 인증 API 계약(docs/api.md)과 DB 스키마(Alembic), 화면의 토큰 처리를 함께 바꾼다. 탭을 닫으면 로그아웃되는 지금의 동작도 바뀐다(쿠키는 탭이 아니라 브라우저 단위). 같은 PC의 HTTP에서 쓰는 지금 배포에서는 `Secure` 쿠키가 localhost에서만 동작한다. 그래서 동작이 바뀌는 설계 결정으로 보고 사용자 결정으로 남긴다

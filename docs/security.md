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

# 보안 점검 기록

이 서비스는 한 PC에서 돌리는 것을 기본으로 한다(DB는 127.0.0.1에만 열리고, 화면·API는 `WEB_HOST_PORT`의 HTTP). 이 문서는 무엇을 확인했고 무엇을 받아들인 위험으로 남겼는지 적는다. 날짜가 붙은 항목은 그날의 결과이며 이후 바뀐 것은 덧붙여 적는다.

## 재현성 고정 (2026-10-15)

같은 파일로 다시 만들었을 때 같은 서버가 뜨도록, 받아 오는 것들을 판 번호나 해시로 고정했다.

2026-10-29 정정: 위 문장이 맞는 범위는 그날 표에 적은 다섯 행(음성 인식 모델, Docker 기본 이미지, 브라우저 말소리 판정 런타임, Python 의존성, npm 의존성)과 표 밖의 비교용 설정 Supertonic 3(리비전과 파일별 SHA-256, `backend/app/services/supertonic.py`)이었다. 서비스가 Hugging Face에서 받는 음성 합성·전처리 모델 — MeloTTS-Korean의 설정·체크포인트, MeloTTS 언어 모듈이 import 때 받는 BERT 토크나이저 6개와 첫 한국어 합성 때 받는 BERT 모델 kykim/bert-kor-base, Kokoro-82M의 설정·가중치·af_heart 목소리 — 는 2026-10-29 전까지 리비전 없이 허브의 `main`을 받고 있었다(고정한 MeloTTS 리비전의 `melo/download_utils.py`·`melo/text/*.py`와 kokoro 0.9.4의 `model.py`·`pipeline.py`가 `revision` 없이 받고, 이 저장소의 합성 코드(`backend/app/services/tts.py`)와 compose에도 이 모델들의 `revision`이나 `HF_HUB_OFFLINE`이 없었다). 그래서 상류 저장소가 바뀐 뒤에는 새로 클론해 빈 `hfcache` 볼륨으로 띄우는 서버뿐 아니라, 네트워크가 있는 채 기존 볼륨으로 다시 시작하는 서버도(huggingface_hub가 `main`이 지금 가리키는 커밋을 허브에 다시 묻는다) README의 수치를 낸 모델과 다른 파일을 조용히 읽을 수 있었다. 2026-10-29에 측정이 읽은 캐시의 커밋으로 고정했다(아래 표의 2026-10-29 행). 측정 수치는 바뀌지 않는다: 합성 후보 측정(docs/experiments.md 3절)의 측정 PC 캐시와 서비스 측정의 `hfcache` 볼륨 모두 저장소마다 스냅숏이 하나뿐이고, 고정한 커밋이 바로 그 스냅숏이다. 아직 고정하지 않은 것도 표 끝에 적었다.

| 대상 | 고정 방법 | 확인 |
|---|---|---|
| 음성 인식 모델 (faster-whisper large-v3-turbo 변환본) | Hugging Face 저장소 커밋 `0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf` (`backend/app/services/stt.py`의 `TURBO_REVISION`, 측정 도구도 같은 값) | 측정에 쓴 PC 캐시와 Docker `hfcache` 볼륨의 스냅숏이 같은 커밋이고, `model.bin`의 SHA-256이 허브의 파일 해시와 같다(`e76620f8…`). 저장소 이름은 그대로 둬 다시 받지 않는다(허브에서는 새 소유자 이름으로 넘어가지만 같은 커밋) |
| Docker 기본 이미지 | 태그와 함께 digest(sha256): `python:3.11-slim-bookworm`, `ghcr.io/astral-sh/uv:0.12`(`backend/Dockerfile`), `node:24-alpine`, `nginx:stable-alpine`(`frontend/Dockerfile`), `postgres:16`, `ollama/ollama:0.35.1`(`docker-compose.yml`, CI의 postgres도 같은 digest) | python 기본 이미지는 고정할 때 이미 빌드돼 있던 api 이미지의 아래 층 세 개와 같은 이미지다(Python 3.11.17). 보안 수정을 받으려면 태그를 다시 받아 digest를 새로 적는다(`backend/Dockerfile` 머리 주석) |
| 브라우저 말소리 판정 런타임 | `onnxruntime-web` 1.30.0 (`frontend/package.json`, 정확한 판) | 1.29.0에서 올리기 전후로 같은 소리를 Edge와 Playwright Chromium에서 돌려 비교했다: FLEURS validation 영어·한국어 세션(조용함, SNR 10dB 분홍 잡음) 15,834조각의 말소리 확률이 두 판에서 비트까지 같았고, Python onnxruntime 1.30.0과의 차이는 최대 1.2×10⁻⁵, 0.5 기준이 바뀐 조각은 0개였다(관문: 1×10⁻⁴ 이하, 뒤집힘 0). 그래서 대화 모드의 말 끝 판정 측정(7절)은 다시 재지 않는다 |
| Python 의존성 | `backend/uv.lock`(CI와 이미지 모두 `--locked`) | — 2026-10-24 덧붙임: 서버가 쓰지 않는 gradio·tensorboard를 잠금에서 빼고 certifi·opentelemetry-api 두 판을 올리면서(아래 '의존성 취약점' 2026-10-24 덧붙임) 잠금을 uv 0.13.1로 다시 썼다(잠금 revision 3 → 5). 이미지가 쓰는 uv 0.12.24도 이 잠금을 그대로 읽는다 |
| npm 의존성 | `frontend/package-lock.json`(`npm ci`) | 아래 '의존성 취약점'의 개발 도구 두 개만 lock에서 올렸고, 빌드 결과(`dist/`)가 올리기 전과 바이트까지 같음을 확인했다. 2026-10-24 덧붙임: 개발 도구(vite 8.3.5 등)를 다시 올리면서 빌드 결과의 자바스크립트 묶음 두 개가 바뀌었다. 새 빌드를 배포 nginx 설정으로 띄워 확인한 내용은 아래 '2026-10-24 덧붙임'에 있다 |
| 음성 합성·전처리 모델 (2026-10-29 추가: MeloTTS-Korean, Kokoro-82M, MeloTTS가 받는 BERT 토크나이저 6개와 한국어 BERT 모델) | `backend/app/services/tts.py`의 `HF_REVISIONS`(Hugging Face 저장소 8개의 커밋)와 `HF_FILES`(서비스 코드가 직접 받는 파일 5개의 SHA-256). MeloTTS-Korean의 `config.json`·`checkpoint.pth`와 Kokoro-82M의 `config.json`·`kokoro-v1_0.pth`·`voices/af_heart.pt`는 고정 커밋으로 받아 SHA-256을 대조한 경로를 MeloTTS(`TTS(config_path=, ckpt_path=)`)와 Kokoro(`KModel(config=, model=)`, 목소리는 `.pt` 파일 경로)에 넘겨 두 라이브러리가 스스로 받지 않게 한다. MeloTTS 언어 모듈이 import 때(토크나이저 6개)와 첫 합성 때(kykim/bert-kor-base 모델) transformers로 받는 BERT는 `AutoTokenizer`·`AutoModelForMaskedLM`의 `from_pretrained`를 MeloTTS를 import하기 전에 감싸 표의 저장소면 `revision=`을 넣는다(표에 없는 저장소는 그대로 넘기고 경고 로그만 남김). SHA-256이 다르면 합성을 끈 채 뜬다(Supertonic과 같은 처리). 커밋 해시로 받으면 huggingface_hub는 캐시에 그 커밋의 파일이 있을 때 허브에 묻지 않고 캐시의 파일을 돌려주므로, 네트워크가 있는 채 다시 시작해도 허브의 새 `main`을 따라가지 않는다. 치르는 값: 시작할 때마다 다섯 파일(합쳐 약 0.5GB)을 한 번 더 읽어 해시하고, 고정한 커밋을 허브에서 더는 받을 수 없게 되면(상류가 이력을 지우는 등) 빈 캐시에서는 다른 커밋으로 넘어가지 않고 합성이 꺼진 채 뜬다 | 측정에 쓴 Docker `hfcache` 볼륨과 측정 PC의 Hugging Face 캐시에서 저장소 8개의 `refs/main`이 모두 같은 커밋이고(MeloTTS-Korean `0207e5ad…`, Kokoro-82M `f3ff3571…`, kykim/bert-kor-base `1779cc09…`, bert-base-uncased `86b5e093…`, bert-base-multilingual-uncased `7cbf9a62…`, dbmdz/bert-base-french-europeana-cased `b895c3cf…`, dccuchile/bert-base-spanish-wwm-uncased `d1c9c456…`, tohoku-nlp/bert-base-japanese-v3 `65243d6e…`) 두 곳 모두 저장소마다 스냅숏이 이것 하나뿐이며, 직접 받는 파일 5개와 kykim/bert-kor-base의 `pytorch_model.bin`의 SHA-256이 두 곳에서 같다(2026-10-29 확인; LFS 파일의 해시는 허브가 캐시 blob 이름으로 쓰는 값과도 같다). 같은 날 허브 API로 본 저장소 8개의 `main`도 모두 이 커밋이었다(그날 새로 클론해도 같은 파일을 받는다). 단위 테스트(`backend/tests/test_tts.py`)가 가짜 transformers·hf_hub_download·melo·kokoro로 revision이 들어가는지, 다른 SHA-256이 거절되는지, 두 번 감싸지 않는지를 확인한다 |
| spaCy 영어 모델 휠과 NLTK 데이터 3개 (2026-10-29 추가) | `backend/Dockerfile`이 받은 파일의 SHA-256을 `sha256sum -c`로 확인한 뒤에만 쓴다(다르거나 받지 못하면 빌드가 멈춘다): `en_core_web_sm-3.8.0-py3-none-any.whl` `1932429d…`, NLTK `averaged_perceptron_tagger_eng.zip` `6025f530…`·`averaged_perceptron_tagger.zip` `e1f13cf2…`·`cmudict.zip` `d07cca47…`(전체 값은 Dockerfile) | 해시는 측정에 쓴 api 이미지(2026-10-15 빌드)의 파일에서 읽었다. NLTK zip 3개는 그 이미지와 2026-10-24에 빌드한 시험 이미지 둘에서 같았고, spaCy 휠은 같은 주소에서 다시 받은 휠 안의 파일 28개가 그 이미지에 설치된 파일과 모두 같았다(GitHub는 이 릴리스 파일의 digest를 주지 않는다). 새 Dockerfile로 두 번 빌드해 두 번 모두 네 해시가 맞았고, 같은 RUN 단계를 해시 하나만 틀리게 해(NLTK zip 하나, 휠 각각) 돌리거나 내려받기를 막으면(네트워크 없음) 그 단계에서 빌드가 멈췄다 |
| 오타 교정 모델 (2026-10-29 추가) | 선택 설정 `CORRECTION_DIGEST`(기본은 확인하지 않음): 주면 Ollama `/api/tags`의 digest가 다를 때 모델을 올리지 않고 교정만 끈다(서버는 뜬다, `backend/app/services/correction.py`). 측정한 빌드 `qwen2.5:1.5b-instruct` = `65ec0654…`(전체 값은 `.env.example`) | docs/experiments.md 16절 '실행 조건'에 적은 digest와, compose `ollama` 볼륨과 측정에 쓴 호스트 Ollama에 있는 이 모델의 manifest 파일 SHA-256이 모두 같다. 두 manifest 파일은 6절 측정 날(2026-09-13)에 받은 뒤로 바뀌지 않았다(파일 시각) |
| 아직 고정하지 않은 것 (2026-10-29 기록) | spaCy 영어 모델 휠과 NLTK 데이터 3개(`backend/Dockerfile`이 해시 없이 받음), 오타 교정 모델(Ollama 태그 `qwen2.5:1.5b-instruct`, digest 확인 없음 — 번역 시험 설정의 `HYMT_DIGEST`와 달리), 측정 후보로만 쓰고 서비스가 쓰지 않는 MeloTTS-English·MMS-TTS(전처럼 리비전 없이 받는다. 고정 코드를 거치는 호출 — MeloTTS-English 적재, 감싼 `AutoTokenizer`로 받는 MMS 토크나이저 — 에서는 경고 로그를 남긴다). 2026-10-29 덧붙임: 같은 날 바로 위 두 행대로 spaCy 휠·NLTK 데이터 3개는 빌드 때 SHA-256으로 확인하게 했고, 오타 교정 모델은 `CORRECTION_DIGEST`를 두면 빌드를 확인한다(기본은 확인하지 않으므로, 그 설정이 없으면 전처럼 태그만 따른다) | — |

## 의존성 취약점 (2026-10-15)

`pip-audit` 2.10.1(PyPI 권고 데이터베이스)과 `npm audit`으로 lock 그대로의 의존성을 검사했다. Python은 서버 환경(gpu + tts 그룹)과 측정 도구 환경(gpu + eval + dev 그룹)을 따로 내보내 검사했다(두 그룹은 함께 설치할 수 없다).

**고친 것** (lock에서 판만 올림, 백엔드 테스트 558개 통과)

| 패키지 | 판 | 권고 | 이 서비스에 해당했나 |
|---|---|---|---|
| PyJWT | 2.14.0 → 2.15.1 | GHSA-42vr-xj54-vc7v (JWKS 흐름에서 깊이 중첩된 내용이 정의되지 않은 예외로 새어 나옴), GHSA-x33g-cr3x-6449 (OKP 비밀 JWK의 공개 키 확인 누락) | 아니다. 이 서비스는 JWKS와 OKP 키를 쓰지 않고, HS256 하나로 서명을 먼저 검증한 뒤에야 내용을 읽으므로 비밀 키 없이는 중첩된 내용을 넣을 수 없다. 인증 경로의 패키지라 올렸다 |
| urllib3 | 2.7.0 → 2.8.0 | GHSA-vxq7-64xx-v4gw·GHSA-gh4c-6fx4-qh6g (악의적 서버의 chunked 응답 스트리밍에서 메모리·무한 반복), GHSA-8988-9cw3-xx77 (HTTPS 프록시 TLS 설정 혼동) | 모델을 처음 받을 때(Hugging Face) 쓰는 경로다. 프록시는 쓰지 않는다 |
| Werkzeug | 3.1.8 → 3.1.9 | GHSA-g6x2-hccm-hh4m (Windows 장치 이름으로 `safe_join` 우회) | 아니다(MeloTTS가 끌어오는 tensorboard의 의존성이고 서비스가 쓰지 않음). 고치는 판이 있어 올렸다. 2026-10-24 덧붙임: tensorboard를 잠금에서 빼면서(아래) werkzeug도 잠금에서 빠졌다 |
| brace-expansion (npm, 개발 도구) | 5.0.9 → 5.0.12 | GHSA-q2hr-2g5m-vwhr 등 3건 (중괄호 확장 DoS) | 아니다(eslint 등 개발 도구 안). 배포 파일에 들어가지 않는다 |
| source-map-js (npm, 개발 도구) | 1.2.1 → 1.2.2 | GHSA-68fv-2mgg-jv7q (색인 소스 맵 DoS) | 아니다(빌드 도구 안) |

**받아들인 것** (고치지 않음, 이유)

| 패키지 | 권고 | 이유 |
|---|---|---|
| transformers 4.27.4 (서버 환경) | 권고 30건(설명 문구를 대강 나눔: 신뢰할 수 없는 체크포인트·원격 코드 역직렬화 약 17건, 토크나이저 정규식 ReDoS 약 10건, 그 밖에 임시 파일·이미지 URL 검증·`save_pretrained` 경로 3건) | MeloTTS가 이 판을 고정해 올릴 수 없다. 서버는 고정된 공개 모델(MeloTTS·BERT 토크나이저)만 읽고, 사용자 입력으로 모델 경로·URL을 정하지 않으며, 토크나이저에 들어가는 글은 번역문(번역 모델이 만든 글)이다. 정규식 ReDoS는 번역문 길이가 입력 제한(글자 입력 500자, 음성 30초)에 묶여 영향이 제한된다(재지 않음). 2026-10-29 덧붙임: '고정된 공개 모델'은 이 판단을 적은 날에는 리비전으로 고정돼 있지 않았다(위 '재현성 고정'의 정정; 지금은 `HF_REVISIONS`로 고정). 또 역직렬화 권고에 대해, 서버가 읽는 pickle 형식 체크포인트에서 안의 코드 실행을 실제로 막는 것은 transformers가 아니라 torch 2.6부터의 `torch.load` 기본값 `weights_only=True`다(서버 코드와 이미지에 이 기본값을 끄는 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` 설정이 없다. 2026-10-29에 서버 이미지의 torch 2.11.0에서, 읽는 순간 코드를 실행하도록 만든 pickle을 같은 호출 모양 `torch.load(파일, map_location="cpu")`로 읽으면 `UnpicklingError`로 거절되고 코드가 실행되지 않음을 확인했다): transformers 4.27.4는 kykim/bert-kor-base의 `pytorch_model.bin`(pickle 형식)을 `torch.load(checkpoint_file, map_location="cpu")`로(`weights_only` 없이) 읽고, MeloTTS도 `checkpoint.pth`를 같은 방식으로 읽는다. 나머지 BERT 토크나이저 파일은 글·JSON 파일(`vocab.txt`, `tokenizer.json` 등)뿐이다. Kokoro 0.9.4는 `weights_only=True`를 직접 준다. 서버 환경(tts 그룹)의 torch가 2.6 아래로 내려가지 않는지 `backend/tests/test_lock_versions.py`가 잠금을 검사한다. 리비전·SHA-256 고정으로 상류 파일이 바뀌어도 서버가 조용히 새 파일을 읽지 않는다 |
| torch 2.11.0+cu128 (서버 환경) | PYSEC-2025-194 (`torch.jit.script` 메모리 손상, 로컬) | 서비스는 `torch.jit.script`를 부르지 않는다. 이 판은 MeloTTS와 CUDA 12.8에 맞춘 tts 그룹 고정이다. pip-audit는 PyPI에 없는 `+cu128` 판을 검사하지 못해 같은 2.11.0을 따로 검사했다 |
| nltk 3.10.3 | PYSEC-2026-3740 (모델 저장·읽기 API의 경로 제한 우회) | 고친 판이 아직 없다. 서비스는 사용자 입력으로 NLTK 모델 경로를 정하지 않는다(이미지에 미리 넣은 데이터만 읽음) |
| setuptools 81.0.0 | PYSEC-2026-3447 (macOS에서 sdist를 만들 때 MANIFEST.in 제외 규칙 우회) | 이 저장소는 sdist를 만들지 않는다. 런타임 경로가 아니다 |

CI의 `audit` 작업이 같은 검사를 push마다 돌린다. 위 '받아들인 것'만 제외하고, 새 권고가 나오면 그 작업이 실패로 표시된다. 처음에는 보고만 하므로(워크플로 전체는 통과) 실패를 보면 이 표에 판단을 더하고 고치거나 받아들인다.

**2026-10-24 덧붙임: 프런트엔드 개발 도구의 새 권고를 고침** (`npm audit --package-lock-only`. 2026-10-20·21에 공개된 vite·vitest 권고와, 그 뒤 같은 검사에 더 나타난 간접 의존성 undici·zod 권고다. `frontend/package.json`에서는 vite·vitest의 하한만 `^8.3.5`·`^5.0.5`로 올렸고(새로 설치해도 취약한 판을 고르지 않게), undici·zod는 상위 패키지(jsdom·eslint-plugin-react-hooks)가 선언한 범위 안에서 lock만 올렸다. 배포 의존성 검사(`npm audit --omit=dev`)는 올리기 전후 모두 0건이라, 배포 파일에 들어가는 라이브러리(react·react-router-dom·onnxruntime-web)는 그대로다)

| 패키지 | 판 | 권고 | 이 서비스에 해당했나 |
|---|---|---|---|
| vite (개발 서버·빌드 도구) | 8.3.0 → 8.3.5 | GHSA-587h-7wmj-h5pm (high, 묶여 들어온 launch-editor의 `fs.existsSync` 존재 확인을 Windows에서 우회), GHSA-698v-4r3x-3vcv (Windows `cmd` 인자에서 `%`를 이스케이프하지 않음), GHSA-r825-w8mm-5pxw (파일 경로가 편집기의 CLI 플래그로 해석됨) — 셋 모두 개발 서버의 `/__open-in-editor` 경로 | **개발 PC에서는 닿을 수 있었다**(배포에는 없음). 개발 서버(`npm run dev`, 그리고 `npm run test:a11y`가 띄우는 Playwright webServer)는 `/__open-in-editor`를 출처 검사 없이 등록하므로, 서버가 떠 있는 동안 개발자가 브라우저로 연 다른 출처의 페이지가 이 경로를 불러 `file` 인자를 넘길 수 있다. 개발 PC가 Windows라 앞의 두 권고를 합치면 편집기가 감지된 상태에서 명령 실행으로 이어질 수 있다(권고 설명 기준, 재현하지 않음). 세 번째는 작업 폴더에 `+`나 `-`로 시작하는 파일과 Vim·Emacs 같은 편집기가 더 있어야 한다. 배포 이미지는 nginx가 빌드 파일만 내주므로 해당 없음 |
| vite | (같음) | GHSA-rq7h-c2jc-7f22 (`server.fs.allow` 밖 파일 노출), GHSA-vfpm-58rq-9qcg (`?vite-wasm-instance`로 `server.fs` 밖 파일의 첫 4바이트 노출), GHSA-9jrq-w75r-8gcw (사용자 정의 미들웨어를 거친 교차 출처 스크립트 실행) | 아니다. 첫째는 Windows가 아닌 기기에서 개발 서버를 `--host`로 네트워크에 열었을 때다. 둘째는 응답 본문을 읽어야 하는 노출인데, 개발 서버는 `localhost:5173`과 Playwright의 `127.0.0.1:15174`에만 열고(`--host`를 쓰지 않음) 개발 서버의 CORS 기본값은 localhost 계열 출처만 허용해 다른 출처의 페이지가 응답을 읽지 못한다. 셋째는 `server.middlewareMode`나 `appType: 'custom'`에서 요청 URL을 그대로 `transformIndexHtml`에 넘길 때인데, 이 저장소는 기본 SPA 설정이고 `frontend/vite.config.ts`의 플러그인은 빌드 전용이다 |
| vitest·@vitest/mocker (단위 테스트 도구) | 5.0.0 → 5.0.5 | GHSA-qxg6-89mg-rqv8 (개발 서버 웹소켓으로 인증 없이 수동 mock을 등록해 코드 주입), GHSA-xp26-3454-mjgp (Windows 역슬래시로 redirect 허용 목록 우회) | 아니다. 두 건의 처리부는 `@vitest/mocker`의 인터셉터 플러그인(브라우저 모드가 Vite 개발 서버의 웹소켓으로 mock 등록을 받는 부분)인데, 이 저장소는 `vitest run`을 jsdom으로만 돌리고 `@vitest/browser`를 설치하지 않아 그 플러그인이 올라가지 않는다 |
| undici (jsdom이 쓰는 HTTP 클라이언트, 개발 도구) | 8.10.2 → 8.12.3 | GHSA-xjj6-f33q-929c (high, content-encoding이 다섯 개를 넘는 응답 뒤 연결이 끊기면 처리되지 않은 오류로 프로세스가 끝남), GHSA-4qv8-2p9j-2685 (EventSource 파서의 메모리 무제한), GHSA-f72v-6w45-q472 (소문자가 아닌 알고리즘 이름으로 SRI 검사 우회) | 아니다. 셋 모두 악의적인 서버의 응답이 있어야 하는데, 단위 테스트는 `fetch`를 가짜로 바꾸고 외부 서버에 접속하지 않는다. 배포 파일과 무관하다 |
| zod (eslint-plugin-react-hooks의 의존성, 개발 도구) | 4.6.2 → 4.6.7 | GHSA-mrhx-c7xc-m3w5 (길이 제한 없는 배열 스키마에 큰 배열을 넣으면 검증 문제 객체가 쌓여 메모리 소진) | 아니다. 린트는 이 저장소의 설정과 소스만 입력으로 받는다. 배포 파일과 무관하다 |

- 확인: `npm audit --package-lock-only` 0건(`--omit=dev`도 0건), `npm ci` 뒤 `npm run lint`·`npm test`(181개)·`npm run build`·`npm run test:a11y`(Microsoft Edge, 8개) 통과. 번들러(rolldown 1.2.8 → 1.2.15)가 함께 올라가 빌드 결과 중 `index-<해시>.js`·`sileroRuntime-<해시>.js`와 그 이름을 가리키는 `index.html`은 올리기 전과 달라졌고, 워클릿·WASM·Silero 모델·CSS·라이선스 원문은 바이트까지 같다. 그래서 새 빌드를 배포 nginx 설정(위 머리글)으로 띄워 Edge로 다시 확인했다: 로그인 → 대화 모드(가짜 마이크 장치, 실제 워클릿·Silero WASM) → 동시통역 → 글자 번역 → 기록 화면을 거치는 동안 CSP 위반 0건·콘솔 오류 0건, 배포된 Silero 청크로 실제 WASM 추론(확률 0.024)과 워클릿 적재 성공, 대조로 `data:` URL 워클릿은 같은 CSP에서 막혔다. API 응답과 웹소켓은 Playwright가 브라우저 안에서 가짜로 답했다(2026-10-15 확인과 달리 API 서버를 띄우지 않음)
- 개발 서버는 `localhost:5173`과 Playwright의 `127.0.0.1:15174`에만 열고(`--host`로 네트워크에 열지 않음) 필요할 때만 띄운다. 개발 도구 권고는 배포 이미지와 무관하지만 개발 PC에 닿는 것이 있으므로, CI `audit` 작업의 npm 결과도 Python 결과와 같이 본다

**2026-10-24 덧붙임: 서버 의존성의 새 권고 — 쓰지 않는 gradio·tensorboard를 잠금에서 빼고, certifi·opentelemetry-api를 올리고, filelock·setuptools는 받아들임** (10-15 뒤 공개된 권고가 2026-10-24 새벽의 CI `audit` 작업(UTC로는 10-23)에서 Python 쪽 39건(pillow 35·brotli 2·certifi 1·opentelemetry-api 1)으로 드러났다. 같은 날 로컬에서 CI와 같은 명령으로 다시 검사하니 서버 환경 45건(pillow 37·brotli 2·certifi 1·opentelemetry-api 1·filelock 3·setuptools 1), 도구 환경 5건(certifi 1·filelock 3·setuptools 1)이었다. 이미 받아들인 nltk·setuptools의 PYSEC 2건은 뺀 수다)

| 조치 | 패키지 | 권고 | 판단 |
|---|---|---|---|
| 잠금에서 뺌 | gradio 6.17.3, tensorboard 2.16.2와 둘만 쓰던 15개(absl-py, brotli, gradio-client, groovy, grpcio, hf-gradio, markdown, orjson, pandas, pillow, safehttpx, semantic-version, tensorboard-data-server, tomlkit, werkzeug) | pillow 12.3.0 37건(이미지 디코더의 메모리 손상·DoS, 고친 판 13.0.0), brotli 1.2.0 PYSEC-2026-4322·GHSA-9v8h-qqgm-4m35(디코더 범위 밖 쓰기, 1.2.1) | MeloTTS가 의존성으로 선언했지만 서버가 쓰는 `melo.api`·`melo.utils`·`melo.text`는 둘을 import하지 않는다(고정한 리비전에서 gradio는 웹 데모 `melo/app.py`, tensorboard는 학습 `melo/train.py`만). 판을 올리는 대신 `backend/pyproject.toml`의 `[tool.uv] exclude-dependencies`로 잠금에서 통째로 빼서 서버 이미지와 검사 대상을 줄였다. override에 `sys_platform == 'never'` 마커를 주는 unidic 방식은 잠금 사본으로 시험하니 설치만 막고 패키지는 잠금에 남아(검사 목록에도 남음) 쓰지 않았다. 되돌아오지 않게 `backend/tests/test_lock_hygiene.py`가 잠금을 검사한다 |
| lock에서 판만 올림 | certifi 2026.7.22 → 2026.10.23 | PYSEC-2026-4320(더 믿지 않기로 한 루트 CA가 묶음에 남아 있음, 고친 판 2026.10.19) | 모델을 처음 받을 때(Hugging Face)의 TLS 검증에 쓰는 루트 묶음이라 올렸다. 두 환경 모두 |
| lock에서 판만 올림 | opentelemetry-api 1.44.0 → 1.45.1 | CVE-2026-114603(`tracestate` 머리글 파싱 ReDoS) | MeloTTS → cached-path → google-cloud-storage → google-api-core가 끌어오는 패키지다. 이 서비스는 tracestate 머리글을 파싱하지 않아 닿지 않지만 고친 판이 있고 google-api-core 2.36.0이 그대로 해석돼 올렸다(서버 환경만 있음) |
| 받아들임(ID로 제외) | filelock 3.32.6 | CVE-2026-112940 / GHSA-v4x8-3wr5-3wq8(POSIX `ReadWriteLock`: 다른 계정이 SQLite 보조 파일을 바꿔 넣음), CVE-2026-112913 / GHSA-9vhv-jxww-xfm5(Windows `ReadWriteLock`: 폴더에 `-wal`·`-journal` 심기), CVE-2026-112907 / GHSA-567q-5qvw-pf92(Windows `SoftFileLock` 계열: 다른 계정의 oplock으로 acquire가 멈춤) — 고친 판 4.2.1만(3.x에는 없음) | 셋 다 잠금 파일이 있는 폴더를 다른 로컬 계정과 공유할 때 성립한다. 이 서비스는 filelock을 직접 쓰지 않고 huggingface_hub·cached-path·torch가 모델 내려받기와 캐시에 `FileLock`을 쓸 뿐이며(`ReadWriteLock`은 쓰지 않음), 그 폴더(컨테이너의 `/models/hf` 볼륨, 측정 PC의 Hugging Face 캐시)는 한 계정만 쓴다. MeloTTS가 고정한 cached-path 1.8.10이 `filelock<4.0`을 요구해 4.2.1로 올릴 수 없다(두 환경 모두 같은 판) |
| 받아들임(ID로 제외) | setuptools 81.0.0(서버 환경)·84.0.0(도구 환경) | GHSA-xmj3-9gf7-h52w(Windows에서 옛 egg 방식 `install_as_egg`로 조작된 wheel을 설치할 때 `namespace_packages.txt` 값으로 폴더 밖에 `__init__.py`를 씀, GitHub 심각도 낮음, 고친 판 84.1.0) | 런타임 경로가 아니다: 서버 이미지는 Linux이고 실행 중 패키지를 설치하지 않으며, 도구 환경은 uv가 잠금에 적힌 wheel만 설치해 setuptools의 egg 설치 경로를 쓰지 않는다. 서버 환경은 torch 2.11.0+cu128이 `setuptools<82`를 요구해 올릴 수 없고, `uv lock --upgrade-package setuptools`는 도구 환경까지 81.0.0으로 맞춰(내려감) 올리지 않았다 |

- 확인: `uv lock --check` 통과, 잠금 diff는 위 17개 제거와 certifi·opentelemetry-api 두 판뿐(잠금 revision 3 → 5, 이미지의 uv 0.12.24도 읽음). CI와 같은 `uv export` → `pip-audit` 2.10.1(위 ID 제외)로 서버·도구 환경 모두 새 권고 0건
- CI `audit` 작업: npm 단계가 Python 단계 실패 뒤에도 돌게 했다(`if: ${{ !cancelled() }}`; 전에는 건너뛰어 npm 결과가 남지 않았다). Actions 탭에서 손으로 돌릴 수 있게 `workflow_dispatch`를 더했다(push 없이 새 권고를 다시 보려고). 주기 실행(schedule)은 저장소에 지속 설정을 두는 일이라 넣지 않았다 — 저장소가 쉬는 동안 공개된 권고는 다음 push나 손으로 돌릴 때 드러난다

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

## 받아들인 위험: 가입 응답으로 계정 존재가 드러남 (2026-10-24 확인, 사용자 결정 대기)

- **지금**: 가입(`POST /api/auth/register`)은 인증 없이 부를 수 있고, 이미 있는 이메일이면 `409`(`Email already registered`, 대소문자만 다른 이메일도 같음), 없으면 `201`로 답한다(docs/api.md의 가입 계약). 로그인은 이메일이 없든 비밀번호가 틀리든 같은 `401`로 답하고, 없는 이메일도 미리 만든 bcrypt 해시(`dummy_hash`)로 같은 검증을 거쳐 응답 시간이 비슷하다(`backend/app/routers/auth.py`)
- **위험**: 화면 포트(`WEB_HOST_PORT`)에 닿는 사람은 이메일을 가입 요청에 넣어 보는 것만으로 그 이메일의 계정이 있는지 알 수 있다. 그래서 로그인의 `dummy_hash` 방어는 로그인 응답 시간으로 알아내는 길만 막고, 같은 정보는 가입 응답으로 바로 드러난다. 가입·로그인 모두 시도 횟수 제한이 없어 이메일 목록을 차례로 넣어 볼 수 있다
- **영향**: 기본 배포는 한 PC에서 쓰는 것이라 작다. 다만 compose는 화면 포트를 호스트 주소 없이 연다(`docker-compose.yml`의 web `ports`. DB 포트는 127.0.0.1에만 열림). 그래서 방화벽이 허용하면 같은 네트워크의 다른 기기도 화면과 `/api`에 닿는다
- **고치지 않은 이유**: 가입 응답을 계정이 있든 없든 같게 만들려면 가입을 확인 메일의 링크로 끝내는 흐름(메일 발송, 확인 전 계정 상태, 이미 가입한 사람에게 보내는 안내)이 필요하다. 이 서비스에는 메일 발송과 이메일 확인 절차가 없다. 응답만 같게 바꾸면 이미 가입한 사람이 다시 가입할 때 성공한 것처럼 보이지만 새 비밀번호로 로그인하지 못해 까닭을 알 수 없다. 인증 API 계약(docs/api.md)과 DB 스키마, 화면의 가입 흐름을 함께 바꾸는 일이라 사용자 결정으로 남긴다. 로그인·가입 시도 제한과, 화면 포트를 이 PC에만 열지도 아직 정하지 않은 설계 결정이다. 같은 네트워크에 열어 쓸 때는 이 셋을 함께 정한다

## API 컨테이너의 실행 사용자 (2026-10-29)

- **바꾼 것**: API 컨테이너는 root가 아닌 `app`(uid·gid 10001)으로 돈다(`backend/Dockerfile`의 `USER app`). 그 전에는 USER 지시가 없어 root(uid 0)로 돌았다. 이 프로세스는 사용자가 올린 임의 형식의 음성 파일을 FFmpeg(PyAV)로 풀고, 위 '의존성 취약점'에서 권고를 받아들인 transformers 4.27.4를 쓴다. 그 안에서 코드가 실행되는 일이 생겨도 컨테이너 안 root 권한을 갖지 않게 하려는 것이다
- **쓸 수 있는 곳**: 모델 캐시 `/models/hf`(hfcache 볼륨), 번역 음성 `/data/audio`(audio 볼륨), HOME `/home/app`(라이브러리의 사용자 캐시), `/tmp`. 가상환경 `/opt/venv`, NLTK 데이터 `/opt/nltk_data`, 서버 코드 `/app`은 root 소유라 이 사용자가 바꿀 수 없다. 새로 만드는 이름 있는 볼륨은 이미지 폴더의 소유권(10001)을 그대로 받고, 이 변경 전의 이미지로 만든 볼륨은 root 소유라 한 번 소유자를 바꾼다(README '전체 실행 (Docker)')
- **확인(GPU 없이, 이미지 안에서)**: `id -u`가 10001이고, 새 볼륨 두 자리와 HOME에는 쓸 수 있고 위 세 곳에는 쓸 수 없다. 이 사용자로 네트워크 없이 NLTK 데이터 찾기, spaCy 영어 모델 적재, MeloTTS 한국어·Kokoro 생성(적재만, 합성 없음)이 된다. NLTK가 받은 zip은 0600으로 만들어져 그대로는 이 사용자가 열지 못했고, 그러면 MeloTTS 영어 전처리(g2p_en)가 import 때마다 데이터를 다시 받으려 해서 이미지에서 읽기 권한을 준다. GPU로 서비스 전체를 띄워 음성 번역 한 건을 저장·재생하는 확인은 아직 하지 않았다
- **남는 위험**: 서비스가 써야 하므로 이 사용자도 hfcache의 모델 파일과 다른 사용자의 번역 음성 파일을 바꾸거나 지울 수 있다. hfcache에서 서비스 코드가 직접 받는 합성 파일 5개는 시작할 때마다 SHA-256을 대조하지만(위 재현성 표), 음성 인식 모델과 BERT 토크나이저·모델은 리비전으로만 고정돼 있어 볼륨 안에서 바뀐 파일은 알아채지 못한다. API 컨테이너는 호스트에 포트를 열지 않고 화면(nginx) 뒤에만 있다

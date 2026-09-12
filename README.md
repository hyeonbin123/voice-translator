# voice-translator

영어↔한국어 음성 번역 웹 서비스. 말하거나 입력하면 음성 인식(STT), 번역, 음성 합성(TTS)을 거쳐 원문과 번역문을 보여 주고 번역문을 읽어 준다. 로그인한 사용자별로 번역 기록을 남긴다.

> 개발 중. 아래는 계획이며, 모델은 공개 평가 데이터(FLEURS)로 측정한 뒤 고른다.

## 구성 (계획)

| 항목 | 내용 |
|---|---|
| 백엔드 | FastAPI(비동기), SQLAlchemy 2.0, PostgreSQL, JWT 인증 |
| 프론트엔드 | React + TypeScript (Vite), 브라우저 마이크 녹음 |
| 음성 인식 | Whisper (faster-whisper, GPU) |
| 번역 | 로컬 번역 모델 후보를 측정해서 선택 (opus-mt, NLLB, 로컬 LLM) |
| 음성 합성 | 한국어·영어를 지원하는 로컬 TTS 후보를 측정해서 선택 |
| 평가 | Google FLEURS 한국어·영어 (음성 인식 CER/WER, 번역 chrF, 지연 시간) |

## 로컬 실행

준비: Python 3.11, [uv](https://docs.astral.sh/uv/), Node.js 24, Docker Desktop. 명령은 모두 프로젝트 폴더에서 시작하고, 백엔드와 프론트엔드는 각각 다른 터미널에서 실행한다.

```bash
# 1. DB (호스트 포트 55442. Windows가 예약하는 5432~5631을 피함)
cp .env.example .env
docker compose up -d db
```

```bash
# 2. 백엔드: http://localhost:8000/api/health
cd backend
uv sync
uv run alembic upgrade head
uv run uvicorn app.main:app --reload
```

API는 `.env`를 자동으로 읽지 않는다. 기본값이 위 compose DB와 같아서 그대로 실행하면 되고, 바꿀 값은 환경 변수로 넘긴다 (`docs/api.md`의 "실행 설정과 DB"). `JWT_SECRET_KEY`를 주지 않으면 서버를 다시 시작할 때마다 로그인이 풀린다.

```bash
# 3. 프론트엔드: http://localhost:5173 (/api 요청은 개발 서버가 백엔드로 넘김)
cd frontend
npm install
npm run dev
```

## 테스트

```bash
(cd backend && uv run ruff check . && uv run pytest)
(cd frontend && npm run lint && npm test && npm run build)
```

백엔드 테스트는 compose DB에 테스트 전용 DB(`vt_test_<임의 문자열>`)를 만들어 쓰고 끝나면 지운다. push와 PR마다 GitHub Actions가 같은 검사를 돌린다 (`.github/workflows/ci.yml`).

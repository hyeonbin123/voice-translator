# 실제 모델을 NVIDIA GPU에서 돌리는 API 이미지(docker-compose.yml이 빌드한다).
# 모델 가중치는 이미지에 넣지 않는다: data/models/ct2(eval/mt_convert.py로 변환한 번역 모델)는 읽기 전용으로
# 연결하고, Whisper·MeloTTS·Kokoro 가중치는 첫 시작 때 hfcache 볼륨으로 받는다. GPU 드라이버는 NVIDIA 컨테이너
# 런타임이 넣어 주고 CUDA 라이브러리는 pip 휠(torch와 gpu 그룹)에서 오므로, CUDA 기본 이미지의 사본을 따로
# 싣지 않는다.
# 기본 이미지는 태그와 함께 digest(sha256)로 고정한다. 태그만 쓰면 같은 Dockerfile이 빌드할 때마다 다른 기본
# 이미지를 받을 수 있어, 측정한 서버와 다시 만든 서버가 달라진다. 이 digest는 2026-10-15에 고정할 때 이미
# 빌드돼 있던 api 이미지의 아래 층과 같은 것을 확인한 것이다(같은 Python 3.11.17). 보안 수정을 받으려면 태그를
# 다시 받아 digest를 새로 적는다: `docker pull python:3.11-slim-bookworm` 뒤
# `docker image inspect python:3.11-slim-bookworm --format '{{index .RepoDigests 0}}'`.
FROM python:3.11-slim-bookworm@sha256:0a310eeecf4e1f5a0743f9a6520c90c88d089c903ca5fd283f501e3a805f5f89

ENV PYTHONUNBUFFERED=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/models/hf \
    NLTK_DATA=/opt/nltk_data

# git: MeloTTS를 저장소에서 설치한다(pyproject.toml에 커밋 고정).
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
# uv 실행 파일만 가져온다. 위 기본 이미지처럼 digest로 고정한다(0.12 태그의 2026-10-15 판).
COPY --from=ghcr.io/astral-sh/uv:0.12@sha256:3af4716e991d6956a41e573eab705d0ee08500cd829ed30293eb8472f372c65a /uv /usr/local/bin/uv

WORKDIR /app
# 의존성을 먼저 설치해 코드만 바뀌었을 때 다시 설치하지 않게 한다. gpu + tts 그룹이 서버 환경이다.
# uv 캐시는 빌드 캐시 마운트에 둔다: 층 안에 남기면 환경 크기가 두 배가 된다.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --group gpu --group tts --no-group dev --no-group eval

# CTranslate2(음성 인식·번역)는 cuBLAS와 cuDNN을 이름으로 찾으므로 휠 안의 폴더를 알려 준다.
ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.11/site-packages/nvidia/cublas/lib:/opt/venv/lib/python3.11/site-packages/nvidia/cudnn/lib

# 글 전처리가 첫 요청 때 받으려 할 데이터를 미리 넣는다(NLTK 품사 태거·발음 사전, spaCy 영어 모델).
RUN --mount=type=cache,target=/root/.cache/uv \
    python -c "import nltk; [nltk.download(r, download_dir='/opt/nltk_data', quiet=True) for r in ('averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger', 'cmudict')]" \
    && uv pip install --python /opt/venv/bin/python \
       https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

COPY alembic.ini ./
COPY alembic ./alembic
COPY app ./app

ENV LOAD_MODELS=true \
    CT2_DIR=/models/ct2 \
    AUDIO_DIR=/data/audio
EXPOSE 8000
# 마이그레이션 뒤 서버를 띄운다. 작업자 하나: 프로세스마다 모델을 한 번씩 올린다(GPU 메모리 약 4.5GB).
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]

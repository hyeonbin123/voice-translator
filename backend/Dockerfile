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
# 받은 파일은 SHA-256을 확인한 뒤에만 쓴다. 둘 다 이름·판 번호로만 받는 파일(NLTK는 nltk_data 저장소의
# gh-pages, spaCy 모델은 GitHub 릴리스)이라, 확인이 없으면 같은 Dockerfile로 다시 빌드해도 다른 파일이 들어올 수
# 있다. 해시는 측정에 쓴 api 이미지(2026-10-15 빌드)에 들어 있던 파일의 값이다: NLTK zip 3개는 그 이미지 안
# 파일의 SHA-256, spaCy 휠은 같은 주소에서 다시 받은 휠로, 안의 파일 28개가 그 이미지에 설치된 파일과 모두
# 같았다(GitHub는 이 릴리스 파일의 digest를 주지 않는다). 다르거나 받지 못하면 `sha256sum -c`가 실패해 빌드가
# 멈춘다(nltk.download는 실패해도 예외 없이 False만 돌려주므로 확인이 이것도 잡는다). 일부러 다른 판을 쓰려면
# 받은 파일의 SHA-256을 새로 적는다.
# - NLTK: nltk.download가 zip을 받아 옆에 풀어 두고 zip도 남긴다. MeloTTS 영어 전처리(g2p_en)는 import 때 zip
#   파일을 찾고(못 찾으면 그 자리에서 내려받으려 함) 태거·사전은 푼 폴더에서 읽는다. nltk 3.10은 받은 파일을
#   0600으로 만들어(nltk/pathsec.py) 아래의 root가 아닌 사용자가 zip을 열지 못하고, 그러면 g2p_en이 시작할 때마다
#   다시 받으려 한다. 그래서 모두 읽을 수 있게(a+rX) 한다.
# - spaCy 휠: 받은 파일을 확인한 뒤 그 파일로 설치하고 지운다(층에 남기지 않음).
RUN --mount=type=cache,target=/root/.cache/uv \
    python -c "import nltk; [nltk.download(r, download_dir='/opt/nltk_data', quiet=True) for r in ('averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger', 'cmudict')]" \
    && printf '%s\n' \
       '6025f530624335c67d6547d44757b357b4e79bae030a0383e9887a92c1718f0b  /opt/nltk_data/taggers/averaged_perceptron_tagger_eng.zip' \
       'e1f13cf2532daadfd6f3bc481a49859f0b8ea6432ccdcd83e6a49a5f19008de9  /opt/nltk_data/taggers/averaged_perceptron_tagger.zip' \
       'd07cca47fd72ad32ea9d8ad1219f85301eeaf4568f8b6b73747506a71fb5afd6  /opt/nltk_data/corpora/cmudict.zip' \
       | sha256sum -c - \
    && chmod -R a+rX /opt/nltk_data \
    && wheel=/tmp/en_core_web_sm-3.8.0-py3-none-any.whl \
    && python -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" \
       https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl \
       "$wheel" \
    && echo "1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85  $wheel" | sha256sum -c - \
    && uv pip install --python /opt/venv/bin/python "$wheel" \
    && rm "$wheel"

# 서버는 root가 아닌 사용자 app(uid·gid 10001)으로 돈다. 이 프로세스는 사용자가 올린 임의 형식의 음성을
# FFmpeg(PyAV)로 풀고 권고를 받아들인 transformers 4.27.4를 쓰므로(docs/security.md), 그 안에서 코드가 실행되는
# 일이 생겨도 root 권한을 갖지 않게 한다. 실행 중 쓰는 곳만 이 사용자에게 준다: 모델 캐시(/models/hf)와 번역
# 음성(/data/audio)은 볼륨이 붙는 자리이고, 새로 만드는 이름 있는 볼륨은 이 폴더의 소유권(10001)을 그대로
# 받는다(이 변경 전의 root 이미지로 만든 볼륨은 README대로 한 번 소유자를 바꾼다). HOME(/home/app)은 torch 등
# 라이브러리의 사용자 캐시가 쓴다. 가상환경(/opt/venv)·NLTK 데이터(/opt/nltk_data)·서버 코드(/app)는 root 소유로
# 남겨 이 사용자가 바꿀 수 없게 한다. uid·gid를 10001로 고정해 README의 `chown -R 10001:10001`과 맞춘다(시스템
# 계정 범위 0~999 밖이라 --system 없이 만든다. --system이면 useradd가 범위를 넘는다고 경고한다). 비밀번호가
# 없어 로그인할 수 없고, --no-log-init은 큰 uid에서 lastlog·faillog 파일이 커지는 것을 막는다.
RUN groupadd --gid 10001 app \
    && useradd --no-log-init --uid 10001 --gid app --create-home --home-dir /home/app \
       --shell /usr/sbin/nologin app \
    && mkdir -p /data/audio /models/hf \
    && chown -R app:app /data/audio /models/hf
ENV HOME=/home/app

COPY alembic.ini ./
COPY alembic ./alembic
COPY app ./app

ENV LOAD_MODELS=true \
    CT2_DIR=/models/ct2 \
    AUDIO_DIR=/data/audio
EXPOSE 8000
USER app
# 마이그레이션 뒤 서버를 띄운다. 작업자 하나: 프로세스마다 모델을 한 번씩 올린다(GPU 메모리 약 4.5GB).
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]

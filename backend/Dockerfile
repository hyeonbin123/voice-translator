# The API with the real models on an NVIDIA GPU (built by docker-compose.yml).
# Model weights are not baked in: data/models/ct2 (the converted translation models, eval/mt_convert.py)
# is mounted read-only, and the Whisper, MeloTTS and Kokoro weights download into the hfcache volume on
# the first start. The GPU driver comes from the NVIDIA container runtime; the CUDA libraries come from the
# pip wheels (torch and the gpu group), so the image doesn't also carry a CUDA base image's copies.
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/models/hf \
    NLTK_DATA=/opt/nltk_data

# git: MeloTTS is installed from its repository (pinned commit in pyproject.toml).
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv

WORKDIR /app
# Dependencies first, so code changes don't reinstall them. gpu + tts is the server environment.
# The uv cache lives in a build cache mount: left in the layer, it doubles the environment's size.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --group gpu --group tts --no-group dev --no-group eval

# CTranslate2 (speech recognition, translation) loads cuBLAS and cuDNN by name; point it at the wheels.
ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.11/site-packages/nvidia/cublas/lib:/opt/venv/lib/python3.11/site-packages/nvidia/cudnn/lib

# Data the text front ends would otherwise fetch on the first request.
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
# Migrate, then serve. One worker: each process loads the models once (about 4.5 GB of GPU memory).
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]

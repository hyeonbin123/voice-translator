"""Speech synthesis (task T4).

The candidates and the selection rule are in docs/experiments.md. Every implementation returns 16-bit
mono PCM WAV (docs/api.md). Model libraries are imported inside the constructors, so importing this
module needs none of them.
"""

# [파일 역할] 음성 합성(TTS) 구현 모음. 서비스가 쓰는 것은 한국어 MeloTTS, 영어 Kokoro-82M과 언어별
# 라우터(LanguageTextToSpeech)이고, MMS-TTS는 측정 후보, Supertonic 3은 비교용 설정이다(docs/experiments.md
# 3·12·14절).
# [설계 이유] 모든 구현이 같은 형식(16비트 모노 PCM WAV)을 돌려줘 API·저장·화면 재생이 모델과 상관없게 한다.
# 모델 라이브러리는 생성자 안에서 import해, 의존성이 서로 충돌하는 후보(예: MeloTTS는 transformers 4.27
# 고정)를 각자 환경에서 측정할 수 있고, tts 그룹을 설치하지 않은 환경에서도 이 모듈을 불러올 수 있다.
# Windows에서만 필요한 작은 우회 두 가지가 들어 있다(README 알려진 한계).
# [연결] services/models.load_speech_synthesis가 서비스용 합성기를 만들고, eval/tts_candidates.py가 측정
# 후보로 쓴다.

from __future__ import annotations

import io
import os
import sys
import types
import wave
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

# numpy: 모델이 내는 float 파형을 16비트 정수 PCM으로 바꾼다.
import numpy as np

from app.services.interfaces import Language, ModelError, SynthesizedAudio, TextToSpeech

T = TypeVar("T")
# MMS 후보의 Hugging Face 저장소 이름과 MeloTTS의 (언어 코드, 화자). 측정에서 언어마다 화자 하나로
# 정했다(docs/experiments.md 3절: MeloTTS는 KR·EN-US).
MMS_REPOS = {"ko": "facebook/mms-tts-kor", "en": "facebook/mms-tts-eng"}
MELO_VOICES = {"ko": ("KR", "KR"), "en": ("EN", "EN-US")}  # language code and speaker in MeloTTS


# float 파형(-1~1)을 WAV 파일 바이트로 감싼다. 범위를 넘는 표본은 잘라(clip) 정수 변환 때 넘침으로 소리가
# 깨지지 않게 하고, 32767을 곱해 little-endian 16비트로 바꾼다. 빈 파형은 실패로 본다. 길이(ms)는 표본 수 ÷
# 표본 추출률로 계산한다.
def wav_bytes(samples: np.ndarray, sample_rate: int) -> SynthesizedAudio:
    """Float samples in [-1, 1] to a 16-bit mono PCM WAV file."""
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        raise ModelError("the model returned no audio")
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(pcm.tobytes())
    duration_ms = round(samples.size * 1000 / sample_rate)
    return SynthesizedAudio(wav=buffer.getvalue(), sample_rate=sample_rate, duration_ms=duration_ms)


# 합성 모델 경계: 어떤 예외든 ModelError로 바꾼다. 파이프라인은 ModelError를 잡아 번역은 돌려주고 tts_error만
# 채운다. 오류 메시지에는 예외 종류만 넣어 번역문이 섞이지 않게 한다.
def _guarded(run: Callable[[], T]) -> T:
    """Any failure inside the model (CUDA errors, text front-end errors) becomes a ModelError."""
    try:
        return run()
    except ModelError:
        raise
    except Exception as exc:  # noqa: BLE001 - the boundary around third-party models
        raise ModelError(f"the speech synthesis model failed: {type(exc).__name__}") from exc


# 이 모델이 다루지 않는 언어이거나 빈 글이면 ModelError.
def _check_text(text: str, language: Language, languages: tuple[Language, ...]) -> None:
    if language not in languages:
        raise ModelError(f"this model does not speak {language}")
    if not text.strip():
        raise ModelError("text is empty")


# 측정 후보(서비스에서 쓰지 않음, 비상업 라이선스). 토크나이저가 uroman을 요구하는 모델(한국어)이면 입력을
# 로마자로 바꾼 뒤 넣는다. torch.no_grad()로 추론만 해 기울기 계산 메모리를 쓰지 않는다.
class MmsTextToSpeech:
    """Meta's MMS-TTS (VITS), one model per language. CC-BY-NC-4.0.

    The Korean model was trained on romanized text, so its input goes through uroman first.
    """

    def __init__(self, language: Language, device: str = "cuda") -> None:
        import torch
        from transformers import AutoTokenizer, VitsModel

        self.language = language
        self.model_name = f"transformers/{MMS_REPOS[language]}"
        self._torch = torch
        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained(MMS_REPOS[language])
        self._model = VitsModel.from_pretrained(MMS_REPOS[language]).to(device).eval()
        self._romanize = None
        if getattr(self._tokenizer, "is_uroman", False):
            from uroman import Uroman

            self._romanize = Uroman().romanize_string

    def synthesize(self, text: str, language: Language) -> SynthesizedAudio:
        _check_text(text, language, (self.language,))

        def run() -> np.ndarray:
            prepared = self._romanize(text) if self._romanize else text
            inputs = self._tokenizer(prepared, return_tensors="pt").to(self._device)
            with self._torch.no_grad():
                return self._model(**inputs).waveform[0].float().cpu().numpy()

        return wav_bytes(_guarded(run), self._model.config.sampling_rate)


# Windows에서 MeloTTS를 import하기 전에 부른다. Linux(Docker)에서는 아무것도 하지 않는다. 일본어 모듈을 빈
# 모듈로 바꿔 끼우되 영어 처리가 쓰는 distribute_phone 하나만 복사해 넣고, g2pkk의 형태소 분석기 확인·생성을
# python-mecab-ko로 바꾼다(monkeypatch).
def _prepare_melo_on_windows() -> None:
    """Work around two MeloTTS import problems on Windows (docs/experiments.md, T4 candidates).

    - melo always imports its Japanese front end, which needs mecab-python3. Its `MeCab` folder is the
      same folder as python-mecab-ko's `mecab` on a case-insensitive file system, and Korean needs the
      latter, so an empty Japanese module stands in (Japanese is never used here).
    - g2pkk looks for `eunjeon` on Windows; python-mecab-ko offers the same pos() call.
    """
    if sys.platform != "win32":
        return
    if "melo.text.japanese" not in sys.modules:
        stub = types.ModuleType("melo.text.japanese")

        def distribute_phone(n_phone: int, n_word: int) -> list[int]:
            # Copied from melo.text.japanese; the English front end imports it from there.
            phones_per_word = [0] * n_word
            for _ in range(n_phone):
                phones_per_word[phones_per_word.index(min(phones_per_word))] += 1
            return phones_per_word

        stub.distribute_phone = distribute_phone  # type: ignore[attr-defined]
        sys.modules["melo.text.japanese"] = stub
    import g2pkk.g2pkk
    import mecab

    g2pkk.g2pkk.G2p.check_mecab = lambda self: None
    g2pkk.g2pkk.G2p.get_mecab = lambda self: mecab.MeCab()


# 서비스의 한국어 합성(측정으로 고름, README: 재인식 한국어 CER 5.58%). tts_to_file에 출력 경로 None을 주면
# 파일을 쓰지 않고 파형 배열을 돌려준다. speed=1.0은 기본 말 속도, quiet=True는 진행 출력을 끈다.
class MeloTextToSpeech:
    """MeloTTS, one model per language. MIT. Pins transformers 4.27, so it needs its own environment."""

    def __init__(self, language: Language, device: str = "cuda") -> None:
        _prepare_melo_on_windows()
        from melo.api import TTS

        code, speaker = MELO_VOICES[language]
        self.language = language
        self.model_name = f"melotts/{code}/{speaker}"
        self._model = TTS(language=code, device=device)
        self._speaker = self._model.hps.data.spk2id[speaker]
        self._sample_rate = self._model.hps.data.sampling_rate

    def synthesize(self, text: str, language: Language) -> SynthesizedAudio:
        _check_text(text, language, (self.language,))
        audio = _guarded(lambda: self._model.tts_to_file(text, self._speaker, None, speed=1.0, quiet=True))
        return wav_bytes(audio, self._sample_rate)


# Windows에서 Kokoro를 올리기 전에 부른다. 이 프로젝트 폴더처럼 경로에 한글이 있으면 espeak-ng가 프로세스를
# 통째로 끝내 버리므로, 작업 폴더를 데이터 폴더의 부모로 옮기고 상대 경로(ASCII)만 넘긴다. 사용자가
# ESPEAK_DATA_PATH를 이미 정했으면 건드리지 않는다.
def _espeak_data_on_windows() -> None:
    """Kokoro's English front end falls back to espeak-ng, whose bundled library cannot open a data path
    with non-ASCII characters on Windows, and ends the whole process when it tries. When the data sits
    under such a path (this project's folder does) and ESPEAK_DATA_PATH isn't set, work from the data's
    parent folder and pass a relative ASCII path. This changes the process's working directory, which is
    safe here because the app uses absolute paths only. Linux containers need nothing.
    """
    if sys.platform != "win32" or os.environ.get("ESPEAK_DATA_PATH"):
        return
    import espeakng_loader

    data = Path(espeakng_loader.get_data_path())
    if str(data).isascii():
        return
    os.chdir(data.parent)
    os.environ["ESPEAK_DATA_PATH"] = data.name


# 서비스의 영어 합성(README: 재인식 영어 WER 3.91%). lang_code="a"는 Kokoro의 미국 영어, 목소리는 af_heart.
# 파이프라인이 글을 조각내 조각마다 파형을 내므로 이어 붙이고, 표본 추출률은 Kokoro의 24kHz로 고정한다.
class KokoroTextToSpeech:
    """Kokoro-82M, English only. Apache-2.0. See _espeak_data_on_windows for its one Windows workaround."""

    def __init__(self, voice: str = "af_heart", device: str = "cuda") -> None:
        _espeak_data_on_windows()
        from kokoro import KPipeline

        self.model_name = f"kokoro/Kokoro-82M/{voice}"
        self._voice = voice
        self._pipeline = KPipeline(lang_code="a", device=device)

    def synthesize(self, text: str, language: Language) -> SynthesizedAudio:
        _check_text(text, language, ("en",))

        def run() -> np.ndarray:
            chunks = [
                audio.cpu().numpy() if hasattr(audio, "cpu") else np.asarray(audio)
                for _, _, audio in self._pipeline(text, voice=self._voice)
            ]
            return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

        return wav_bytes(_guarded(run), 24_000)


# 비교용 설정(KO_TTS=supertonic). 실제 ONNX 추론은 services/supertonic.SupertonicEngine이 맡고, 이 클래스는
# TextToSpeech 인터페이스와 WAV 변환만 맡는다. engine을 직접 넘길 수 있어 테스트가 가짜 엔진을 끼운다.
# model_name에 단계 수와 GPU 여부를 넣어 기록과 /api/health에서 어떤 설정으로 합성했는지 구분한다. providers는
# 세션별로 실제로 올라간 ONNX Runtime 제공자 목록이다(측정 기록용).
class SupertonicTextToSpeech:
    """Supertonic 3 (Supertone) on ONNX Runtime: the CPU provider uses no GPU memory (T78, docs/experiments.md
    12); the CUDA provider is a trial that needs the onnxruntime-gpu package (T82, docs/experiments.md 14).

    The model is under OpenRAIL-M, whose use restrictions bind the service and its users (README). The
    upstream project is archived, so the files are pinned by revision and SHA-256 and read from a local copy
    (app/services/supertonic.py, eval/supertonic_download.py). It speaks Korean here; Kokoro keeps English.
    """

    def __init__(
        self,
        model_dir: Path | None = None,
        *,
        steps: int = 8,
        threads: int = 2,
        voice: str | None = None,
        languages: tuple[Language, ...] = ("ko",),
        verify: bool = True,
        provider: str = "cpu",
        engine=None,
    ) -> None:
        from app.services import supertonic

        voice = voice or supertonic.DEFAULT_VOICE
        self.languages = languages
        self.steps = steps
        # The CPU name stays what T78 recorded; the GPU says so (it shows in /api/health).
        self.model_name = f"supertonic-3/{voice}/{steps}-step" + ("/cuda" if provider == "cuda" else "")
        if engine is None:
            if model_dir is None:
                raise ValueError("model_dir is needed to load Supertonic")
            engine = supertonic.SupertonicEngine.load(
                model_dir, voice=voice, threads=threads, verify=verify, provider=provider
            )
        self._engine = engine
        self.providers: dict[str, list[str]] = dict(getattr(engine, "providers", {}))

    def synthesize(self, text: str, language: Language) -> SynthesizedAudio:
        _check_text(text, language, self.languages)
        audio = _guarded(lambda: self._engine.synthesize(text, language, self.steps))
        return wav_bytes(audio, self._engine.sample_rate)


# 언어마다 다른 합성기로 보내는 라우터(한국어 MeloTTS, 영어 Kokoro). model_name은 "ko: ..., en: ..." 형태로
# 기록과 /api/health에 나간다. 등록되지 않은 언어면 ModelError.
class LanguageTextToSpeech:
    """Routes each language to its own model, since the best model may differ by language."""

    def __init__(self, by_language: dict[Language, TextToSpeech]) -> None:
        self._by_language = by_language
        self.model_name = ", ".join(f"{lang}: {model.model_name}" for lang, model in by_language.items())

    def synthesize(self, text: str, language: Language) -> SynthesizedAudio:
        model = self._by_language.get(language)
        if model is None:
            raise ModelError(f"no speech synthesis model for {language}")
        return model.synthesize(text, language)

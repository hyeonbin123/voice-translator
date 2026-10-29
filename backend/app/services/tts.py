"""음성 합성.

후보와 선택 규칙은 docs/experiments.md에 있다. 모든 구현은 16비트 모노 PCM WAV를 돌려준다(docs/api.md).
모델 라이브러리는 생성자 안에서 import하므로, 이 모듈을 불러올 때는 그 라이브러리가 하나도 필요 없다.
서비스가 Hugging Face에서 받는 합성·전처리 모델은 측정에 쓴 커밋(리비전)으로 고정해 받는다
(HF_REVISIONS·HF_FILES, docs/security.md '재현성 고정').
"""

# [파일 역할] 음성 합성(TTS) 구현 모음. 서비스가 쓰는 것은 한국어 MeloTTS, 영어 Kokoro-82M과 언어별
# 라우터(LanguageTextToSpeech)이고, MMS-TTS는 측정 후보, Supertonic 3은 비교용 설정이다(docs/experiments.md
# 3·12·14절). 서비스 모델이 Hugging Face에서 받는 파일을 측정에 쓴 커밋과 SHA-256으로 고정하는 부분
# (HF_REVISIONS·HF_FILES, pinned_file, pin_transformers_revisions)도 여기 있다.
# [설계 이유] 모든 구현이 같은 형식(16비트 모노 PCM WAV)을 돌려줘 API·저장·화면 재생이 모델과 상관없게 한다.
# 모델 라이브러리는 생성자 안에서 import해, 의존성이 서로 충돌하는 후보(예: MeloTTS는 transformers 4.27
# 고정)를 각자 환경에서 측정할 수 있고, tts 그룹을 설치하지 않은 환경에서도 이 모듈을 불러올 수 있다.
# Windows에서만 필요한 작은 우회 두 가지가 들어 있다(README 알려진 한계).
# [연결] services/models.load_speech_synthesis가 서비스용 합성기를 만들고, eval/tts_candidates.py가 측정
# 후보로 쓴다.

from __future__ import annotations

import hashlib
import io
import logging
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

logger = logging.getLogger(__name__)

T = TypeVar("T")
# MMS 후보의 Hugging Face 저장소 이름과 MeloTTS의 (언어 코드, 화자). 측정에서 언어마다 화자 하나로
# 정했다(docs/experiments.md 3절: MeloTTS는 KR·EN-US).
MMS_REPOS = {"ko": "facebook/mms-tts-kor", "en": "facebook/mms-tts-eng"}
MELO_VOICES = {"ko": ("KR", "KR"), "en": ("EN", "EN-US")}
# MeloTTS 언어 코드 → 설정·체크포인트가 있는 Hugging Face 저장소(고정 MeloTTS melo/download_utils.py의
# LANG_TO_HF_REPO_ID와 같은 값). Kokoro는 저장소 하나에 설정·가중치·목소리가 함께 있다.
MELO_REPOS = {"KR": "myshell-ai/MeloTTS-Korean", "EN": "myshell-ai/MeloTTS-English"}
KOKORO_REPO = "hexgrad/Kokoro-82M"
KOKORO_WEIGHTS = "kokoro-v1_0.pth"

# [합성·전처리 모델의 리비전 고정] 서비스가 Hugging Face에서 받는 저장소 → 커밋 해시. 값은 측정이 읽은 두
# 캐시, 곧 README의 합성 수치를 낸 후보 측정(docs/experiments.md 3절)의 측정 PC 캐시와 서비스 측정의 컨테이너
# hfcache 볼륨의 `refs/main`이다. 두 캐시 모두 저장소마다 스냅숏이 이것 하나뿐이고 값이 같다(2026-10-29 확인,
# docs/security.md '재현성 고정'). 고정 MeloTTS와 kokoro 0.9.4는 revision 없이 받으므로
# (hf_hub_download·from_pretrained에 revision 인자가 없음) 그대로 두면 상류 저장소가 바뀐 뒤 시작하는 서버(빈
# 캐시로 새로 띄우거나, 네트워크가 있는 채 다시 시작해 허브가 main을 다시 알려 줄 때)는 측정과 다른 모델을
# 조용히 읽는다. 커밋 해시로 받으면 huggingface_hub는 캐시에 그 커밋의 파일이 있을 때 허브에 묻지 않고 바로
# 돌려준다(0.36.2 file_download.py의 commit hash 지름길).
# - MeloTTS-Korean: 서비스 한국어 합성의 config.json·checkpoint.pth.
# - Kokoro-82M: 서비스 영어 합성의 config.json·kokoro-v1_0.pth·voices/af_heart.pt.
# - BERT 6개: 고정 MeloTTS의 언어 모듈이 `import melo.api` 때 모듈 수준에서 AutoTokenizer.from_pretrained로
#   받는 토크나이저(README '음성 합성 준비물'). kykim/bert-kor-base는 첫 한국어 합성 때
#   AutoModelForMaskedLM으로 모델(pytorch_model.bin)도 받는다(melo/text/japanese_bert.py get_bert_feature).
# 표에 없는 저장소(MeloTTS-English·MMS 측정 후보, ZH 전용 BERT)는 고정하지 않고 전처럼 받는다(거절하지 않고,
# 아래 고정 함수를 거치는 호출에서만 경고 로그를 남긴다).
HF_REVISIONS: dict[str, str] = {
    "myshell-ai/MeloTTS-Korean": "0207e5adfc90129a51b6b03d89be6d84360ed323",
    "hexgrad/Kokoro-82M": "f3ff3571791e39611d31c381e3a41a3af07b4987",
    "kykim/bert-kor-base": "1779cc0982ada0216dd6de0dd4e86fb78201926d",
    "bert-base-uncased": "86b5e0934494bd15c9632b12f734a8a67f723594",
    "bert-base-multilingual-uncased": "7cbf9a625e29989f6b9c6c2fa68234c304f7e38f",
    "dbmdz/bert-base-french-europeana-cased": "b895c3cf291f7bf4c15639078a6bee0b3e272c5b",
    "dccuchile/bert-base-spanish-wwm-uncased": "d1c9c4565c9d6731e57ed7f027b802697bad861e",
    "tohoku-nlp/bert-base-japanese-v3": "65243d6e5629b969c77309f217bd7b1a79d43c7e",
}
# 이 모듈이 직접 받는 파일의 (저장소, 파일 이름) → SHA-256. 커밋 해시가 같아도 캐시가 손상되거나 다른 파일이
# 들어오면 적재가 멈추도록 Supertonic(app/services/supertonic.py)과 같은 방식으로 받을 때마다 대조한다. 값은
# 측정에 쓴 hfcache 볼륨과 PC 캐시의 파일을 2026-10-29에 해시한 것으로, 큰 파일(LFS)은 허브가 캐시 blob
# 이름으로 쓰는 SHA-256과도 같다. transformers가 스스로 받는 BERT 파일은 여기 없고 위 커밋으로만 고정한다.
# 대조 때문에 시작할 때마다 이 파일들(합쳐 약 0.5GB)을 한 번 더 읽는다.
HF_FILES: dict[tuple[str, str], str] = {
    ("myshell-ai/MeloTTS-Korean", "config.json"): (
        "74543376976dfadde45ba34336fa79c7e95509f43a7c2e701b22c0f71fd7695c"
    ),
    ("myshell-ai/MeloTTS-Korean", "checkpoint.pth"): (
        "48e3ff3fd0b5348e095f0468e60ae727507564100f58142ef3a922ead6e0a4d0"
    ),
    ("hexgrad/Kokoro-82M", "config.json"): "5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f",
    ("hexgrad/Kokoro-82M", "kokoro-v1_0.pth"): (
        "496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4"
    ),
    ("hexgrad/Kokoro-82M", "voices/af_heart.pt"): (
        "0ab5709b8ffab19bfd849cd11d98f75b60af7733253ad0d67b12382a102cb4ff"
    ),
}


def wav_bytes(samples: np.ndarray, sample_rate: int) -> SynthesizedAudio:
    """[-1, 1] 범위의 float 표본을 16비트 모노 PCM WAV 파일로 만든다.

    범위를 넘는 표본은 잘라(clip) 정수 변환 때 넘침으로 소리가 깨지지 않게 하고, 32767을 곱해
    little-endian 16비트로 바꾼다. 빈 파형은 실패로 본다. 길이(ms)는 표본 수 ÷ 표본 추출률로 계산한다.
    """
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


def _guarded(run: Callable[[], T]) -> T:
    """모델 안의 어떤 실패든(CUDA 오류, 글 전처리 오류) ModelError로 바꾼다.

    번역 흐름은 ModelError를 잡아 번역은 돌려주고 tts_error만 채운다. 오류 메시지에는 예외 종류만 넣어
    번역문이 섞이지 않게 한다.
    """
    try:
        return run()
    except ModelError:
        raise
    except Exception as exc:  # noqa: BLE001 - 외부 모델을 감싸는 예외 경계
        raise ModelError(f"the speech synthesis model failed: {type(exc).__name__}") from exc


# 이 모델이 다루지 않는 언어이거나 빈 글이면 ModelError.
def _check_text(text: str, language: Language, languages: tuple[Language, ...]) -> None:
    if language not in languages:
        raise ModelError(f"this model does not speak {language}")
    if not text.strip():
        raise ModelError("text is empty")


# 파일의 SHA-256(16진수). 큰 체크포인트(수백 MB)를 한꺼번에 메모리에 올리지 않도록 1MiB씩 읽는다.
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pinned_file(repo_id: str, filename: str) -> Path:
    """HF_FILES에 고정한 파일 하나를 고정 커밋에서 받아(캐시에 있으면 그대로) SHA-256을 확인한 경로를
    돌려준다.

    huggingface_hub.hf_hub_download에 HF_REVISIONS의 커밋을 revision으로 준다. 커밋 해시를 주면 캐시(HF_HOME,
    컨테이너에서는 hfcache 볼륨)에 그 커밋의 파일이 있을 때 허브에 묻지 않으므로, 네트워크가 없어도 뜨고
    상류가 바뀌어도 다시 받지 않는다. 받은(또는 캐시의) 파일을 HF_FILES의 SHA-256과 대조해 다르면 ModelError로
    멈춘다: services/models.load_speech_synthesis가 이것을 잡아 합성만 끄고 서버는 뜬다(docs/api.md
    tts_error). 표에 없는 파일은 해시를 확인할 수 없으므로 받지 않고 거절한다. 입력: 저장소 이름과 저장소 안
    경로("voices/af_heart.pt"처럼 폴더가 들어갈 수 있음). 출력: 캐시 안 실제 파일 경로(Path).
    """
    expected = HF_FILES.get((repo_id, filename))
    if expected is None:
        raise ModelError(f"{repo_id}/{filename} is not pinned to a SHA-256")
    # huggingface_hub는 주 의존성(faster-whisper)이 끌어오지만, 테스트가 바꿔 끼울 수 있게 여기서 import한다.
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(repo_id, filename, revision=HF_REVISIONS[repo_id]))
    if _sha256(path) != expected:
        raise ModelError(f"{repo_id}/{filename} does not match its pinned SHA-256")
    return path


def pin_transformers_revisions() -> None:
    """transformers의 AutoTokenizer·AutoModelForMaskedLM.from_pretrained를 감싸, HF_REVISIONS의 저장소를 읽을
    때 그 커밋을 revision으로 넣는다. 프로세스 동안 유지되고 두 번 불러도 한 번만 감싼다.

    고정 MeloTTS는 언어 모듈을 import하는 순간(melo/text/english.py·french.py·spanish.py·japanese.py·
    korean.py·chinese_mix.py의 모듈 수준 `AutoTokenizer.from_pretrained(model_id)`)과 첫 합성 때
    (melo/text/japanese_bert.py get_bert_feature의 `AutoModelForMaskedLM.from_pretrained(model_id)`) revision
    없이 BERT를 받는다. 두 호출 모두 MeloTTS 안에 있어 인수를 바꿀 수 없으므로, `from melo.api import TTS`보다
    먼저 이 함수를 불러 클래스의 from_pretrained를 바꿔 둔다. 생성자 안에서만 걸었다 풀면 첫 합성 때의 모델
    적재를 놓치므로 되돌리지 않는다.
    - 표에 있는 저장소: 호출자가 revision을 주지 않았을 때만 표의 커밋을 넣는다(직접 준 revision은 그대로).
    - 표에 없는 허브 저장소(예: MMS 측정 후보 facebook/mms-tts-kor): 거절하지 않고 그대로 넘기되 경고 로그를
      남긴다. 로컬 폴더 경로(eval/mt_convert.py처럼 스냅숏 폴더를 주는 경우)는 허브 저장소가 아니라 경고도
      없다.
    - 감싼 함수는 staticmethod로 둔다: 클래스로 부르든 인스턴스로 부르든 같은 함수가 불리고, 원래
      classmethod는 감쌀 때 이미 그 클래스에 묶어 두었다. 감싼 표시(_pinned_revisions)로 멱등을 지킨다.
    transformers 4.27.4(MeloTTS가 고정한 판)와 5.x 모두 두 from_pretrained가 revision 키워드 인자를 받는다.
    """
    import transformers

    for cls in (transformers.AutoTokenizer, transformers.AutoModelForMaskedLM):
        if getattr(cls.from_pretrained, "_pinned_revisions", None) is HF_REVISIONS:
            continue
        cls.from_pretrained = staticmethod(_pinning(cls.__name__, cls.from_pretrained))


# cls_name의 from_pretrained(원래 함수, 클래스에 묶인 classmethod)를 감싼 함수를 만든다.
# pin_transformers_revisions의 반복문 밖에 두어 각 클래스가 자기 원래 함수를 붙잡게 한다(반복 변수를 닫아 두면
# 마지막 클래스 것만 남는다).
def _pinning(cls_name: str, original: Callable) -> Callable:
    def from_pretrained(*args, **kwargs):
        name = args[0] if args else kwargs.get("pretrained_model_name_or_path")
        if kwargs.get("revision") is None and isinstance(name, str | os.PathLike):
            name = os.fspath(name)
            revision = HF_REVISIONS.get(name)
            if revision is not None:
                kwargs["revision"] = revision
            elif not os.path.isdir(name):
                logger.warning(
                    "%s.from_pretrained(%r) is not pinned to a revision; the hub's current main is used",
                    cls_name,
                    name,
                )
        return original(*args, **kwargs)

    from_pretrained._pinned_revisions = HF_REVISIONS  # type: ignore[attr-defined]
    return from_pretrained


# 다른 코드(예: 측정 도구)가 pin_transformers_revisions보다 먼저 melo.api를 import했으면, 언어 모듈의
# 토크나이저는 이미 revision 없이 받아진 뒤다. 고칠 수는 없으므로(모듈 수준 호출이 끝남) 경고만 남겨 로그에서
# 알 수 있게 한다. 같은 프로세스에서 MeloTextToSpeech를 두 번째 만들 때는 첫 번째가 먼저 감싸 두었으므로
# (표시가 있음) 경고하지 않는다.
def _warn_if_melo_was_imported_unpinned() -> None:
    transformers = sys.modules.get("transformers")
    if "melo.api" not in sys.modules or transformers is None:
        return
    from_pretrained = getattr(getattr(transformers, "AutoTokenizer", None), "from_pretrained", None)
    if getattr(from_pretrained, "_pinned_revisions", None) is None:
        logger.warning(
            "melo.api was imported before the revision pins; its BERT tokenizers were fetched without a "
            "revision"
        )


# MeloTTS 언어 코드에 고정 저장소가 있으면, 고정 커밋에서 받아 SHA-256을 확인한 config.json·checkpoint.pth
# 경로를 TTS(config_path=, ckpt_path=)에 넘길 인수로 돌려준다(고정 MeloTTS melo/api.py가 두 인자를 받고,
# 주어지면 download_utils가 허브에 묻지 않는다). 표에 없는 언어(MeloTTS-English 측정 후보)는 빈 dict를
# 돌려주고 경고를 남겨 전처럼 melo가 스스로 받게 한다.
def _melo_pinned_paths(code: str) -> dict[str, str]:
    repo = MELO_REPOS.get(code)
    if repo is None or (repo, "checkpoint.pth") not in HF_FILES:
        logger.warning("MeloTTS %s (%s) is not pinned; melo downloads the hub's current main", code, repo)
        return {}
    return {
        "config_path": str(pinned_file(repo, "config.json")),
        "ckpt_path": str(pinned_file(repo, "checkpoint.pth")),
    }


class MmsTextToSpeech:
    """Meta의 MMS-TTS(VITS). 언어마다 모델 하나. CC-BY-NC-4.0.

    한국어 모델은 로마자로 바꾼 글로 학습됐으므로 입력을 먼저 uroman에 통과시킨다(토크나이저가 uroman을
    요구할 때). 측정 후보이며(비상업 라이선스) 서비스에서는 쓰지 않는다. torch.no_grad()로 추론만 해 기울기
    계산 메모리를 쓰지 않는다.
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


def _prepare_melo_on_windows() -> None:
    """Windows에서 MeloTTS import의 두 가지 문제를 우회한다(docs/experiments.md 3절의 합성 후보).

    MeloTTS를 import하기 전에 부른다. Linux(Docker)에서는 아무것도 하지 않는다.
    - melo는 늘 일본어 전처리를 import하는데, 이것은 mecab-python3가 필요하다. 그 `MeCab` 폴더는 대소문자를
      구분하지 않는 파일 시스템에서 python-mecab-ko의 `mecab` 폴더와 같은 폴더가 되고, 한국어에는 뒤의
      것이 필요하다. 그래서 빈 일본어 모듈을 대신 끼운다(여기서는 일본어를 쓰지 않는다). 영어 처리가 쓰는
      distribute_phone 하나만 복사해 넣는다.
    - g2pkk는 Windows에서 `eunjeon`을 찾는다. python-mecab-ko가 같은 pos() 호출을 제공하므로 g2pkk의
      형태소 분석기 확인·생성을 python-mecab-ko로 바꾼다(monkeypatch).
    """
    if sys.platform != "win32":
        return
    if "melo.text.japanese" not in sys.modules:
        stub = types.ModuleType("melo.text.japanese")

        def distribute_phone(n_phone: int, n_word: int) -> list[int]:
            # melo.text.japanese에서 그대로 복사했다. 영어 전처리가 거기서 이 함수를 import한다.
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


class MeloTextToSpeech:
    """MeloTTS. 언어마다 모델 하나. MIT. transformers 4.27을 고정하므로 자기만의 환경이 필요하다.

    서비스의 한국어 합성이다(측정으로 고름, README: 재인식 한국어 CER 5.58%). tts_to_file에 출력 경로
    None을 주면 파일을 쓰지 않고 파형 배열을 돌려준다. speed=1.0은 기본 말 속도, quiet=True는 진행 출력을
    끈다. 한국어(KR)의 설정·체크포인트는 고정 커밋에서 받아 SHA-256을 확인한 경로로 넘기고, MeloTTS가 받는
    BERT 토크나이저·모델은 pin_transformers_revisions로 커밋을 고정한다(docs/security.md '재현성 고정').
    체크포인트(pickle)는 melo가 `torch.load(ckpt_path, map_location=device)`로 읽으므로, 안의 코드 실행을 막는
    것은 torch 2.6부터의 기본값 weights_only=True다(tests/test_lock_versions.py가 잠금의 torch 판을 지킨다).
    """

    def __init__(self, language: Language, device: str = "cuda") -> None:
        _prepare_melo_on_windows()
        # melo.api를 import하면 언어 모듈들이 모듈 수준에서 BERT 토크나이저를 받으므로, 그보다 먼저 감싼다.
        _warn_if_melo_was_imported_unpinned()
        pin_transformers_revisions()
        from melo.api import TTS

        code, speaker = MELO_VOICES[language]
        self.language = language
        self.model_name = f"melotts/{code}/{speaker}"
        self._model = TTS(language=code, device=device, **_melo_pinned_paths(code))
        self._speaker = self._model.hps.data.spk2id[speaker]
        self._sample_rate = self._model.hps.data.sampling_rate

    def synthesize(self, text: str, language: Language) -> SynthesizedAudio:
        _check_text(text, language, (self.language,))
        audio = _guarded(lambda: self._model.tts_to_file(text, self._speaker, None, speed=1.0, quiet=True))
        return wav_bytes(audio, self._sample_rate)


def _espeak_data_on_windows() -> None:
    """Windows에서 espeak-ng가 ASCII가 아닌 데이터 경로 때문에 프로세스를 끝내지 않게 한다.

    Kokoro의 영어 전처리는 espeak-ng로 넘어가는데, 그 묶음 라이브러리는 Windows에서 ASCII가 아닌 글자가
    든 데이터 경로를 열지 못하고, 열려고 하면 프로세스 전체를 끝내 버린다. 데이터가 그런 경로 아래에
    있고(이 프로젝트 폴더가 그렇다) ESPEAK_DATA_PATH가 정해져 있지 않으면, 데이터의 부모 폴더로 작업
    폴더를 옮기고 상대 ASCII 경로를 넘긴다. 프로세스의 작업 폴더를 바꾸지만 앱은 절대 경로만 쓰므로
    여기서는 안전하다. 사용자가 ESPEAK_DATA_PATH를 이미 정했으면 건드리지 않는다. Linux 컨테이너는 아무것도
    할 필요가 없다. Kokoro를 올리기 전에 부른다.
    """
    if sys.platform != "win32" or os.environ.get("ESPEAK_DATA_PATH"):
        return
    import espeakng_loader

    data = Path(espeakng_loader.get_data_path())
    if str(data).isascii():
        return
    os.chdir(data.parent)
    os.environ["ESPEAK_DATA_PATH"] = data.name


class KokoroTextToSpeech:
    """Kokoro-82M, 영어 전용. Apache-2.0. Windows 우회 하나는 _espeak_data_on_windows를 본다.

    서비스의 영어 합성이다(README: 재인식 영어 WER 3.91%). lang_code="a"는 Kokoro의 미국 영어, 목소리는
    af_heart. 파이프라인이 글을 조각내 조각마다 파형을 내므로 이어 붙이고, 표본 추출률은 Kokoro의
    24kHz로 고정한다. 설정·가중치·목소리는 고정 커밋에서 받아 SHA-256을 확인한 경로로 넘긴다(docs/security.md
    '재현성 고정'): kokoro 0.9.4의 KModel(config=, model=)은 경로가 주어지면 허브에 묻지 않고, 파이프라인은
    `.pt`로 끝나는 voice를 파일 경로로 읽는다(둘 다 torch.load(..., weights_only=True)).
    """

    def __init__(self, voice: str = "af_heart", device: str = "cuda") -> None:
        _espeak_data_on_windows()
        from kokoro import KModel, KPipeline

        self.model_name = f"kokoro/Kokoro-82M/{voice}"
        # KPipeline에 만들어 둔 KModel을 주면 device 인자를 쓰지 않으므로, 파이프라인이 스스로 만들 때와 같이
        # 여기서 장치로 옮기고 eval로 둔다.
        model = KModel(
            repo_id=KOKORO_REPO,
            config=str(pinned_file(KOKORO_REPO, "config.json")),
            model=str(pinned_file(KOKORO_REPO, KOKORO_WEIGHTS)),
        )
        model = model.to(device).eval()
        self._pipeline = KPipeline(lang_code="a", repo_id=KOKORO_REPO, model=model, device=device)
        # 고정한 목소리는 파일 경로로 넘긴다(kokoro는 ','로 여러 목소리를 섞을 수 있어 이름을 쪼개지만 캐시
        # 경로에는 ','가 없다). 표에 없는 목소리는 이름 그대로 넘겨 kokoro가 받게 하고 경고를 남긴다.
        voice_file = f"voices/{voice}.pt"
        if (KOKORO_REPO, voice_file) in HF_FILES:
            self._voice = str(pinned_file(KOKORO_REPO, voice_file))
        else:
            logger.warning("Kokoro voice %s is not pinned; kokoro downloads the hub's current main", voice)
            self._voice = voice

    def synthesize(self, text: str, language: Language) -> SynthesizedAudio:
        _check_text(text, language, ("en",))

        def run() -> np.ndarray:
            chunks = [
                audio.cpu().numpy() if hasattr(audio, "cpu") else np.asarray(audio)
                for _, _, audio in self._pipeline(text, voice=self._voice)
            ]
            return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

        return wav_bytes(_guarded(run), 24_000)


class SupertonicTextToSpeech:
    """ONNX Runtime으로 돌리는 Supertonic 3(Supertone). 비교용 설정(KO_TTS=supertonic)이다.

    CPU 제공자는 GPU 메모리를 쓰지 않고(docs/experiments.md 12절), CUDA 제공자는 onnxruntime-gpu 패키지가
    필요한 시험 설정이다(docs/experiments.md 14절). 모델은 OpenRAIL-M 라이선스이고 그 사용 제한은 서비스와
    사용자에게도 적용된다(README). 상위 프로젝트가 보관 처리돼 파일을 리비전과 SHA-256으로 고정하고 로컬
    사본에서 읽는다(app/services/supertonic.py, eval/supertonic_download.py). 여기서는 한국어를 맡고 영어는
    Kokoro가 그대로 맡는다. 실제 ONNX 추론은 services/supertonic.SupertonicEngine이 맡고, 이 클래스는
    TextToSpeech 인터페이스와 WAV 변환만 맡는다. engine을 직접 넘길 수 있어 테스트가 가짜 엔진을 끼운다.
    providers는 세션별로 실제로 올라간 ONNX Runtime 제공자 목록이다(측정 기록용).
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
        # model_name에 단계 수와 GPU 여부를 넣어 기록과 /api/health에서 어떤 설정으로 합성했는지 구분한다.
        # CPU 이름은 CPU 시험이 기록한 그대로 두고, GPU면 "/cuda"를 붙인다.
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


class LanguageTextToSpeech:
    """언어마다 그 언어의 모델로 보낸다. 가장 좋은 모델이 언어마다 다를 수 있기 때문이다.

    서비스는 한국어 MeloTTS와 영어 Kokoro를 묶는다. model_name은 "ko: ..., en: ..." 형태로 기록과
    /api/health에 나간다. 등록되지 않은 언어면 ModelError.
    """

    def __init__(self, by_language: dict[Language, TextToSpeech]) -> None:
        self._by_language = by_language
        self.model_name = ", ".join(f"{lang}: {model.model_name}" for lang, model in by_language.items())

    def synthesize(self, text: str, language: Language) -> SynthesizedAudio:
        model = self._by_language.get(language)
        if model is None:
            raise ModelError(f"no speech synthesis model for {language}")
        return model.synthesize(text, language)

# [파일 역할] 음성 합성 서비스(app/services/tts.py)에서 모델 없이 시험할 수 있는 부분을 검증한다: 파형 → WAV
# 변환(wav_bytes), 모델 실패를 ModelError로 바꾸는 경계(_guarded), 입력 검사(_check_text), 언어별 모델
# 연결(LanguageTextToSpeech), 그리고 합성·전처리 모델을 측정에 쓴 Hugging Face 커밋으로 고정하는 부분
# (pin_transformers_revisions, pinned_file, MeloTextToSpeech·KokoroTextToSpeech가 고정한 경로를 모델
# 라이브러리에 넘기는지), MeloTTS 한국어 전처리(g2pkk)의 정규식 캐시(_CompiledRegexCache·_cache_g2pkk_regex)가
# re와 같은 결과를 내고 끼우기·되돌리기가 정확한지. 실제 g2pkk로 켜기 전후 출력을 견주는 테스트는
# tests/test_g2pkk_regex_cache.py에 따로 있다(g2pkk가 있을 때만 돎).
# [왜 중요한가] 합성 모델(MeloTTS·Kokoro 등)마다 출력 형식과 실패 방식이 다르다. 이 도우미들이 서비스 쪽에 늘
# 같은 WAV 형식과 같은 오류 종류(ModelError)를 주어야, 합성이 실패해도 번역은 돌려주는 처리(README 설계
# 판단)가 동작한다. 리비전 고정은 상류 저장소가 바뀌어도 README에 기록한 수치를 낸 모델과 같은 파일을 읽게
# 하는 장치라(docs/security.md '재현성 고정'), 고정한 저장소에 revision이 들어가고 다른 저장소는 그대로
# 지나가며 SHA-256이 다른 파일은 거절하는지를 모델 라이브러리 없이(가짜 transformers·hf_hub_download·
# melo.api·kokoro) 확인한다.

import hashlib
import io
import logging
import re
import sys
import types
import wave
from types import SimpleNamespace

import numpy as np
import pytest

# _guarded·_check_text는 모듈 내부 도우미지만 동작을 직접 고정하려고 가져온다. FakeTextToSpeech는 단어 수 ×
# 100 ms 길이의 무음을 만드는 가짜 합성기다. tts 모듈 자체도 가져와 monkeypatch로 Windows 우회·고정 파일
# 함수를 바꿔 끼운다.
from app.services import tts
from app.services.interfaces import ModelError
from app.services.tts import (
    HF_FILES,
    HF_REVISIONS,
    KokoroTextToSpeech,
    LanguageTextToSpeech,
    MeloTextToSpeech,
    _check_text,
    _guarded,
    pin_transformers_revisions,
    pinned_file,
    wav_bytes,
)
from tests.fakes import FakeTextToSpeech


# float 파형을 16비트 모노 PCM WAV로 쓰는지, [-1, 1]을 넘는 값(2.0, -2.0)은 잘라서 ±32767이 되는지(정수
# 넘침으로 부호가 뒤집히지 않게), 0.5는 16383(0.5 × 32767을 정수로 자름)인지 확인한다. 길이는 ms 단위로
# 반올림해 5표본은 0 ms다.
def test_wav_bytes_writes_16_bit_mono_pcm_and_clips():
    audio = wav_bytes(np.array([0.0, 0.5, -0.5, 2.0, -2.0], dtype=np.float32), 16_000)
    with wave.open(io.BytesIO(audio.wav)) as clip:
        assert (clip.getnchannels(), clip.getsampwidth(), clip.getframerate()) == (1, 2, 16_000)
        samples = np.frombuffer(clip.readframes(clip.getnframes()), dtype="<i2")
    assert samples.tolist() == [0, 16383, -16383, 32767, -32767]
    assert audio.sample_rate == 16_000
    assert audio.duration_ms == 0  # 16 kHz에서 5표본은 반올림하면 0ms


# 표본화율이 다른 모델(24 kHz)도 길이를 표본 수 ÷ 표본화율로 맞게 계산하는지 본다.
def test_wav_bytes_reports_the_duration():
    assert wav_bytes(np.zeros(24_000), 24_000).duration_ms == 1000


# 모델이 빈 파형을 돌려주면 빈 WAV를 저장하지 않고 ModelError로 실패시킨다.
def test_empty_audio_is_a_model_error():
    with pytest.raises(ModelError):
        wav_bytes(np.zeros(0), 16_000)


# 모델 안의 어떤 예외(예: CUDA 메모리 부족 RuntimeError)도 ModelError로 바뀌고, 메시지에는 원래 예외의 종류만
# 남는지 확인한다. 성공하면 반환값을 그대로 넘긴다.
def test_guarded_turns_engine_failures_into_model_errors():
    def crash():
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(ModelError, match="RuntimeError"):
        _guarded(crash)
    assert _guarded(lambda: 42) == 42


# 공백뿐인 글자와 모델이 지원하지 않는 언어는 모델을 부르기 전에 ModelError로 거절한다.
def test_text_and_language_are_checked():
    with pytest.raises(ModelError):
        _check_text("   ", "ko", ("ko",))
    with pytest.raises(ModelError, match="does not speak ko"):
        _check_text("안녕", "ko", ("en",))


# LanguageTextToSpeech는 언어마다 다른 모델(한국어 MeloTTS, 영어 Kokoro)을 쓰려고 둔 연결 계층이다. 해당 언어
# 모델로 넘기는지(두 단어 → 200 ms), model_name이 '언어: 모델' 형식인지(/api/health에 그대로 보임), 모델이
# 없는 언어는 ModelError인지 확인한다.
def test_language_router_uses_each_model_and_rejects_missing_languages():
    router = LanguageTextToSpeech({"en": FakeTextToSpeech()})
    assert router.synthesize("hello there", "en").duration_ms == 200
    assert router.model_name == "en: fake-tts"
    with pytest.raises(ModelError, match="no speech synthesis model for ko"):
        router.synthesize("안녕", "ko")


# ---- 합성·전처리 모델의 리비전 고정 (docs/security.md '재현성 고정' 2026-10-29 정정) ----


# 고정 표 자체의 모양: 커밋 해시는 40자리 16진수, SHA-256은 64자리 16진수이고, 파일 해시를 둔 저장소는 모두
# 리비전 표에도 있어야 한다(리비전 없이 해시만 맞추면 허브에 매번 묻게 된다). 서비스가 받는 다섯 파일이 빠지지
# 않았는지도 이름으로 고정한다.
def test_the_pin_tables_are_well_formed_and_cover_the_service_files():
    for repo, revision in HF_REVISIONS.items():
        assert len(revision) == 40 and int(revision, 16) >= 0, repo
    for (repo, filename), digest in HF_FILES.items():
        assert repo in HF_REVISIONS, (repo, filename)
        assert len(digest) == 64 and int(digest, 16) >= 0, (repo, filename)
    assert set(HF_FILES) >= {
        ("myshell-ai/MeloTTS-Korean", "config.json"),
        ("myshell-ai/MeloTTS-Korean", "checkpoint.pth"),
        ("hexgrad/Kokoro-82M", "config.json"),
        ("hexgrad/Kokoro-82M", "kokoro-v1_0.pth"),
        ("hexgrad/Kokoro-82M", "voices/af_heart.pt"),
    }
    # MeloTTS 언어 모듈이 import 때 받는 BERT 토크나이저 6개(README '음성 합성 준비물')와 한국어 BERT 모델.
    assert set(HF_REVISIONS) >= {
        "kykim/bert-kor-base",
        "bert-base-uncased",
        "bert-base-multilingual-uncased",
        "dbmdz/bert-base-french-europeana-cased",
        "dccuchile/bert-base-spanish-wwm-uncased",
        "tohoku-nlp/bert-base-japanese-v3",
    }


# 가짜 transformers 모듈: AutoTokenizer·AutoModelForMaskedLM이 진짜처럼 classmethod from_pretrained를 갖고,
# 받은 인수를 calls에 적는다. 실제 transformers 4.27.4도 두 클래스의 from_pretrained가 classmethod다.
def fake_transformers(calls: list) -> types.ModuleType:
    class FakeAuto:
        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
            calls.append((cls.__name__, pretrained_model_name_or_path, args, dict(kwargs)))
            return (cls.__name__, pretrained_model_name_or_path)

    class AutoTokenizer(FakeAuto):
        pass

    class AutoModelForMaskedLM(FakeAuto):
        pass

    module = types.ModuleType("transformers")
    module.AutoTokenizer = AutoTokenizer  # type: ignore[attr-defined]
    module.AutoModelForMaskedLM = AutoModelForMaskedLM  # type: ignore[attr-defined]
    return module


# 감싼 from_pretrained의 핵심 동작: 표의 저장소에는 그 커밋이 revision으로 들어가고(다른 인수는 그대로
# 지나감), 표에 없는 저장소는 revision 없이 그대로 넘어가되 경고 로그가 남고(같은 모듈의 MMS 측정 후보가 쓰는
# facebook/mms-tts-kor를 거절하면 안 됨), 호출자가 revision을 직접 주면 바꾸지 않으며, 로컬 폴더 경로는 허브
# 저장소가 아니라서 경고도 내지 않는다. 반환값은 원래 함수의 것이어야 한다.
def test_pinned_repositories_get_their_revision_and_others_pass_through(monkeypatch, caplog, tmp_path):
    calls: list = []
    module = fake_transformers(calls)
    monkeypatch.setitem(sys.modules, "transformers", module)
    pin_transformers_revisions()

    result = module.AutoTokenizer.from_pretrained("kykim/bert-kor-base")
    assert result == ("AutoTokenizer", "kykim/bert-kor-base")
    assert calls[-1] == (
        "AutoTokenizer",
        "kykim/bert-kor-base",
        (),
        {"revision": HF_REVISIONS["kykim/bert-kor-base"]},
    )

    module.AutoModelForMaskedLM.from_pretrained("bert-base-uncased", output_hidden_states=True)
    assert calls[-1] == (
        "AutoModelForMaskedLM",
        "bert-base-uncased",
        (),
        {"output_hidden_states": True, "revision": HF_REVISIONS["bert-base-uncased"]},
    )

    with caplog.at_level(logging.WARNING, logger="app.services.tts"):
        module.AutoTokenizer.from_pretrained("facebook/mms-tts-kor")
    assert calls[-1] == ("AutoTokenizer", "facebook/mms-tts-kor", (), {})
    assert "facebook/mms-tts-kor" in caplog.text

    module.AutoTokenizer.from_pretrained("kykim/bert-kor-base", revision="other")
    assert calls[-1][3] == {"revision": "other"}

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="app.services.tts"):
        module.AutoTokenizer.from_pretrained(str(tmp_path))
    assert calls[-1] == ("AutoTokenizer", str(tmp_path), (), {})
    assert caplog.text == ""


# 두 번 불러도 한 번만 감싼다: MeloTextToSpeech를 여러 번 만들거나(측정 도구) 다른 언어 모듈이 또 부르더라도
# 감싸기가 겹쳐 쌓이지 않아야 한다. 겹치면 호출마다 로그가 중복되고 되돌리기도 어려워진다. 두 번째 호출 뒤에도
# from_pretrained 객체가 같고, 한 번 부를 때 원래 함수가 한 번만 불린다.
def test_pinning_transformers_twice_wraps_once(monkeypatch):
    calls: list = []
    module = fake_transformers(calls)
    monkeypatch.setitem(sys.modules, "transformers", module)
    pin_transformers_revisions()
    wrapped = module.AutoTokenizer.from_pretrained
    pin_transformers_revisions()
    assert module.AutoTokenizer.from_pretrained is wrapped
    module.AutoTokenizer.from_pretrained("kykim/bert-kor-base")
    assert len(calls) == 1


# 고정 파일 받기: hf_hub_download에 저장소·파일 이름과 표의 커밋이 revision으로 들어가고, 받은 파일의
# SHA-256이 표와 같을 때만 경로를 돌려준다. 내용이 다르면(상류가 바뀌었거나 캐시가 손상됨) ModelError로 멈춰
# 서버가 다른 모델을 조용히 읽지 않게 하고, 표에 없는 파일은 해시를 확인할 수 없으므로 받지도 않고 거절한다.
def test_pinned_file_passes_the_revision_and_checks_the_sha_256(monkeypatch, tmp_path):
    good = tmp_path / "config.json"
    good.write_bytes(b'{"a": 1}')
    seen: dict = {}

    def fake_download(repo_id, filename, **kwargs):
        seen.update(repo_id=repo_id, filename=filename, **kwargs)
        return str(good)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    monkeypatch.setitem(HF_REVISIONS, "fake/repo", "0" * 40)
    monkeypatch.setitem(HF_FILES, ("fake/repo", "config.json"), hashlib.sha256(b'{"a": 1}').hexdigest())

    assert pinned_file("fake/repo", "config.json") == good
    assert seen == {"repo_id": "fake/repo", "filename": "config.json", "revision": "0" * 40}

    good.write_bytes(b'{"a": 2}')
    with pytest.raises(ModelError, match="SHA-256"):
        pinned_file("fake/repo", "config.json")

    seen.clear()
    with pytest.raises(ModelError, match="not pinned"):
        pinned_file("fake/repo", "other.bin")
    assert seen == {}  # 표에 없는 파일은 받지도 않는다


# MeloTextToSpeech 생성자 흐름을 가짜 melo.api로 확인한다: ① transformers 감싸기를 melo.api를 import하기 전에
# 부른다(언어 모듈이 import 때 토크나이저를 받으므로 순서가 바뀌면 고정이 빠진다) ② 한국어(KR)는 고정 커밋에서
# 받아 SHA-256을 확인한 config.json·checkpoint.pth 경로를 TTS(config_path=, ckpt_path=)로 넘겨 melo가 스스로
# 받지 않게 한다 ③ 표에 없는 MeloTTS-English(측정 후보)는 전처럼 TTS(language, device)만 부르고 경고를 남긴다.
# melo.api 모듈의 TTS는 PEP 562 __getattr__로 주어 `from melo.api import TTS`가 일어난 순간이 events에 남는다.
def fake_melo(monkeypatch, events: list, hps_sampling_rate: int = 44_100):
    class FakeTTS:
        def __init__(self, language, device="auto", use_hf=True, config_path=None, ckpt_path=None):
            events.append(("TTS", language, device, config_path, ckpt_path))
            data = SimpleNamespace(spk2id={"KR": 0, "EN-US": 3}, sampling_rate=hps_sampling_rate)
            self.hps = SimpleNamespace(data=data)

        def tts_to_file(self, text, speaker_id, output_path=None, speed=1.0, quiet=False):
            events.append(("tts_to_file", text, speaker_id, output_path, speed, quiet))
            return np.zeros(hps_sampling_rate // 10, dtype=np.float32)

    api = types.ModuleType("melo.api")

    def __getattr__(name):
        if name == "TTS":
            events.append(("import melo.api",))
            return FakeTTS
        raise AttributeError(name)

    api.__getattr__ = __getattr__  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "melo", types.ModuleType("melo"))
    monkeypatch.setitem(sys.modules, "melo.api", api)
    monkeypatch.setattr(tts, "_prepare_melo_on_windows", lambda: None)
    monkeypatch.setattr(tts, "pin_transformers_revisions", lambda: events.append(("pin",)))


def test_melo_korean_loads_the_pinned_checkpoint_after_pinning_transformers(monkeypatch, tmp_path):
    events: list = []
    fake_melo(monkeypatch, events)
    files = {
        ("myshell-ai/MeloTTS-Korean", "config.json"): tmp_path / "config.json",
        ("myshell-ai/MeloTTS-Korean", "checkpoint.pth"): tmp_path / "checkpoint.pth",
    }
    monkeypatch.setattr(tts, "pinned_file", lambda repo, filename: files[(repo, filename)])

    model = MeloTextToSpeech("ko", device="cpu")

    assert events == [
        ("pin",),
        ("import melo.api",),
        ("TTS", "KR", "cpu", str(tmp_path / "config.json"), str(tmp_path / "checkpoint.pth")),
    ]
    assert model.model_name == "melotts/KR/KR"
    audio = model.synthesize("안녕하세요", "ko")
    assert events[-1] == ("tts_to_file", "안녕하세요", 0, None, 1.0, True)
    assert (audio.sample_rate, audio.duration_ms) == (44_100, 100)


# 측정 도구처럼 다른 코드가 먼저 `import melo.api`를 했으면 언어 모듈의 토크나이저는 이미 revision 없이 받아진
# 뒤라, 생성자는 경고를 남긴다. 반대로 이 모듈이 먼저 감싸 둔 뒤(표시가 있음)라면 — 같은 프로세스에서 두 번째
# 인스턴스를 만드는 경우 — 경고하지 않는다. 가짜 transformers를 sys.modules에 두어 실제 라이브러리 없이 본다.
def test_melo_warns_when_melo_api_was_imported_before_the_pins(monkeypatch, caplog):
    events: list = []
    fake_melo(monkeypatch, events)
    monkeypatch.setattr(tts, "pinned_file", lambda repo, filename: f"/pinned/{filename}")
    calls: list = []
    transformers = fake_transformers(calls)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    assert sys.modules["melo.api"].TTS is not None  # 측정 도구가 먼저 import한 것처럼 melo.api를 건드린다

    with caplog.at_level(logging.WARNING, logger="app.services.tts"):
        MeloTextToSpeech("ko", device="cpu")
    assert "imported before the revision pins" in caplog.text

    caplog.clear()
    pinned = tts._pinning("AutoTokenizer", lambda *args, **kwargs: None)
    transformers.AutoTokenizer.from_pretrained = staticmethod(pinned)
    with caplog.at_level(logging.WARNING, logger="app.services.tts"):
        MeloTextToSpeech("ko", device="cpu")
    assert "imported before the revision pins" not in caplog.text


def test_melo_english_candidate_is_not_pinned_and_warns(monkeypatch, caplog):
    events: list = []
    fake_melo(monkeypatch, events)
    monkeypatch.setattr(tts, "pinned_file", lambda repo, filename: pytest.fail("no pinned file for EN"))

    with caplog.at_level(logging.WARNING, logger="app.services.tts"):
        model = MeloTextToSpeech("en", device="cpu")

    assert events == [("pin",), ("import melo.api",), ("TTS", "EN", "cpu", None, None)]
    assert model.model_name == "melotts/EN/EN-US"
    assert "MeloTTS-English" in caplog.text


# regex_cache=True(설정 MELO_REGEX_CACHE)면 생성자가 g2pkk 정규식 캐시를 `from melo.api import TTS`보다 먼저
# 켠다. 기본(False)은 부르지 않아 생성 순서가 이 설정이 없던 때와 같다. 캐시를 켜도 모델 이름은
# 그대로다(출력이 같으므로 기록·/api/health에서 다른 모델로 보이면 안 된다). 실제 g2pkk 대신 호출 순서만
# 남기는 함수를 끼운다.
def test_melo_turns_on_the_g2pkk_regex_cache_before_importing_melo(monkeypatch, tmp_path):
    events: list = []
    fake_melo(monkeypatch, events)
    monkeypatch.setattr(tts, "pinned_file", lambda repo, filename: tmp_path / filename)

    def cache_g2pkk_regex():
        events.append(("regex cache",))
        return ("g2pkk.g2pkk",), lambda: None

    monkeypatch.setattr(tts, "_cache_g2pkk_regex", cache_g2pkk_regex)

    model = MeloTextToSpeech("ko", device="cpu", regex_cache=True)
    assert events == [
        ("regex cache",),
        ("pin",),
        ("import melo.api",),
        ("TTS", "KR", "cpu", str(tmp_path / "config.json"), str(tmp_path / "checkpoint.pth")),
    ]
    assert model.model_name == "melotts/KR/KR"

    events.clear()
    MeloTextToSpeech("ko", device="cpu")
    assert ("regex cache",) not in events


# 여기부터 g2pkk 정규식 캐시의 대리 객체(_CompiledRegexCache). g2pkk가 쓰는 re.sub·re.findall과 같은 인수를
# 받아 같은 결과를 내야 한다: 역참조 치환(\1, g2pkk special.py의 꼴), 지우기, count, flags, 이미 컴파일된
# 패턴, 함수 치환, bytes. re 쪽은 count·flags를 키워드로 넘긴다(위치 인수는 Python 3.13부터 경고).
@pytest.mark.parametrize(
    ("pattern", "repl", "string", "count", "flags"),
    [
        ("([ᄌᄍᄎ])ᅧ", r"\1ᅥ", "져쪄", 0, 0),
        ("/[PJEB]", "", "나/J 는/E 가/P", 0, 0),
        ("a", "b", "aaaa", 2, 0),
        ("A", "x", "aAa", 0, re.IGNORECASE),
        (re.compile("(b)"), r"[\1]", "abcb", 0, 0),
        (r"(?P<word>\w+)", lambda match: match.group("word").upper(), "ab cd", 0, 0),
        (b"(a)", rb"\1\1", b"cab", 0, 0),
    ],
)
def test_regex_cache_sub_matches_re(pattern, repl, string, count, flags):
    cache = tts._CompiledRegexCache(maxsize=8)
    expected = re.sub(pattern, repl, string, count=count, flags=flags)
    assert cache.sub(pattern, repl, string, count, flags) == expected
    assert cache.sub(pattern, repl, string, count=count, flags=flags) == expected
    assert cache.sub(pattern, repl, string, count=count, flags=flags) == expected  # 캐시에서 꺼낸 두 번째


# findall도 같다: 그룹이 없으면 문자열 목록(english.py의 영어 단어 찾기), 그룹이 둘이면 튜플 목록
# (numerals.py의 '숫자+단위/B' 찾기), flags, 컴파일된 패턴.
@pytest.mark.parametrize(
    ("pattern", "string", "flags"),
    [
        ("[A-Za-z']+", "I'm sending an mp3 file", 0),
        (r"([\d][\d,]*)([ㄱ-힣]+)/B", "3개/B를 1,000원/B에", 0),
        ("A", "aAa", re.IGNORECASE),
        (re.compile("b+"), "abbcb", 0),
    ],
)
def test_regex_cache_findall_matches_re(pattern, string, flags):
    cache = tts._CompiledRegexCache(maxsize=8)
    expected = re.findall(pattern, string, flags=flags)
    assert cache.findall(pattern, string, flags) == expected
    assert cache.findall(pattern, string, flags=flags) == expected


# 잘못된 입력에는 re와 같은 예외를 낸다(컴파일된 패턴에 flags를 주면 ValueError, 틀린 정규식은 re.error).
# sub·findall 말고는 모두 진짜 re로 넘긴다: 함수(compile·match·escape)와 상수·예외 클래스가 re의 것과 같다.
def test_regex_cache_raises_like_re_and_passes_everything_else_to_re():
    cache = tts._CompiledRegexCache(maxsize=8)
    with pytest.raises(ValueError, match="compiled pattern"):
        cache.sub(re.compile("a"), "b", "a", flags=re.IGNORECASE)
    with pytest.raises(re.error):
        cache.findall("(", "x")
    assert cache.compile is re.compile and cache.match is re.match and cache.escape is re.escape
    assert cache.IGNORECASE is re.IGNORECASE and cache.error is re.error and cache.Pattern is re.Pattern
    assert cache.match("a", "abc").group() == "a"


# (패턴, flags)마다 한 번만 컴파일해 다시 쓰고(두 번째부터는 적중), 서로 다른 패턴이 상한을 넘으면 오래 안 쓴
# 것부터 버려 크기가 상한에서 멈춘다(입력에 따라 패턴이 무한히 늘어나도 메모리가 묶인다).
def test_regex_cache_compiles_each_pattern_once_and_stays_bounded():
    cache = tts._CompiledRegexCache(maxsize=4)
    for _ in range(3):
        cache.sub("a(b)", r"\1", "ab")
        cache.findall("b", "abb")
    info = cache.cache_info()
    assert (info.misses, info.hits, info.currsize, info.maxsize) == (2, 4, 2, 4)

    for index in range(10):
        cache.sub(f"x{index}", "", "x1")
    assert cache.cache_info().currsize == 4


# g2pkk 다섯 모듈(g2pkk.g2pkk·special·utils·english·numerals)과 부모 패키지 자리에 가짜 모듈을 끼우고 {이름:
# 모듈}을 돌려준다. 기본은 모두 진짜 re를 갖고, without_re의 모듈은 re 속성이 없으며(g2pkk regular.py처럼),
# other_re의 모듈은 다른 코드가 이미 바꿔 끼운 다른 객체를 갖는다. monkeypatch가 sys.modules를 되돌린다.
def fake_g2pkk(monkeypatch, *, without_re=(), other_re=()):
    monkeypatch.setitem(sys.modules, "g2pkk", types.ModuleType("g2pkk"))
    modules = {}
    for name in ("g2pkk.g2pkk", "g2pkk.special", "g2pkk.utils", "g2pkk.english", "g2pkk.numerals"):
        module = types.ModuleType(name)
        if name in other_re:
            module.re = SimpleNamespace()
        elif name not in without_re:
            module.re = re
        monkeypatch.setitem(sys.modules, name, module)
        modules[name] = module
    return modules


# 다섯 모듈의 re를 모두 대리 객체 하나로 바꾸고 그 이름을 돌려준다. 다시 불러도 대리 객체를 또 감싸지 않으며
# (멱등) 그 호출은 끼운 것이 없어 이름이 비고 되돌리기도 아무것도 하지 않는다. 첫 호출의 되돌리기가 모두 진짜
# re로 돌려놓고, 두 번 불러도 괜찮다.
def test_cache_g2pkk_regex_swaps_re_in_the_five_modules_and_restores(monkeypatch):
    modules = fake_g2pkk(monkeypatch)

    names, restore = tts._cache_g2pkk_regex()
    assert names == ("g2pkk.g2pkk", "g2pkk.special", "g2pkk.utils", "g2pkk.english", "g2pkk.numerals")
    assert all(module.re is tts._G2PKK_RE for module in modules.values())

    again, restore_again = tts._cache_g2pkk_regex()
    assert again == ()
    restore_again()
    assert all(module.re is tts._G2PKK_RE for module in modules.values())

    restore()
    assert all(module.re is re for module in modules.values())
    restore()
    assert all(module.re is re for module in modules.values())


# 진짜 re를 가진 모듈만 바꾼다: re가 없는 모듈은 그대로 두고, 다른 코드가 이미 바꿔 끼운 re도 덮어쓰지 않는다.
# 되돌리기는 자기가 끼운 대리 객체가 아직 그 자리에 있을 때만 진짜 re로 돌려놓는다(그 뒤 다른 코드가 바꾼 것은
# 그대로 둔다).
def test_cache_g2pkk_regex_only_replaces_the_real_re(monkeypatch):
    modules = fake_g2pkk(monkeypatch, without_re=("g2pkk.numerals",), other_re=("g2pkk.utils",))
    other = modules["g2pkk.utils"].re

    names, restore = tts._cache_g2pkk_regex()
    assert names == ("g2pkk.g2pkk", "g2pkk.special", "g2pkk.english")
    assert not hasattr(modules["g2pkk.numerals"], "re")
    assert modules["g2pkk.utils"].re is other

    replaced = SimpleNamespace()
    modules["g2pkk.english"].re = replaced
    restore()
    assert modules["g2pkk.english"].re is replaced
    assert modules["g2pkk.g2pkk"].re is re and modules["g2pkk.special"].re is re


# KokoroTextToSpeech 생성자 흐름을 가짜 kokoro로 확인한다: 설정·가중치는 고정 커밋에서 받아 SHA-256을 확인한
# 경로를 KModel(repo_id, config=, model=)에 넘기고(kokoro가 스스로 받지 않음), 그 모델을 device로 옮겨 eval로
# 두고 KPipeline(lang_code="a", repo_id, model=)에 준다. 목소리는 af_heart.pt의 고정 경로를 파이프라인 호출의
# voice로 넘겨(kokoro는 .pt로 끝나는 voice를 파일로 읽음) 허브에서 받지 않게 한다. model_name은 전과 같다.
def fake_kokoro(monkeypatch, events: list):
    class FakeKModel:
        def __init__(self, repo_id=None, config=None, model=None, disable_complex=False):
            events.append(("KModel", repo_id, config, model))

        def to(self, device):
            events.append(("to", device))
            return self

        # torch.nn.Module.eval()(추론 모드 전환)을 흉내 낸 메서드다. 내장 함수 eval과는 상관없다.
        def eval(self):
            events.append(("eval",))
            return self

    class FakeKPipeline:
        def __init__(self, lang_code, repo_id=None, model=True, trf=False, en_callable=None, device=None):
            events.append(("KPipeline", lang_code, repo_id, model, device))

        def __call__(self, text, voice=None, speed=1):
            events.append(("call", text, voice))
            yield ("graphemes", "phonemes", np.zeros(2_400, dtype=np.float32))

    module = types.ModuleType("kokoro")
    module.KModel = FakeKModel  # type: ignore[attr-defined]
    module.KPipeline = FakeKPipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kokoro", module)
    monkeypatch.setattr(tts, "_espeak_data_on_windows", lambda: None)
    return FakeKModel


def test_kokoro_loads_the_pinned_weights_and_voice_file(monkeypatch, tmp_path):
    events: list = []
    FakeKModel = fake_kokoro(monkeypatch, events)
    files = {
        ("hexgrad/Kokoro-82M", "config.json"): tmp_path / "config.json",
        ("hexgrad/Kokoro-82M", "kokoro-v1_0.pth"): tmp_path / "kokoro-v1_0.pth",
        ("hexgrad/Kokoro-82M", "voices/af_heart.pt"): tmp_path / "voices" / "af_heart.pt",
    }
    monkeypatch.setattr(tts, "pinned_file", lambda repo, filename: files[(repo, filename)])

    model = KokoroTextToSpeech(device="cpu")

    assert events[:3] == [
        ("KModel", "hexgrad/Kokoro-82M", str(tmp_path / "config.json"), str(tmp_path / "kokoro-v1_0.pth")),
        ("to", "cpu"),
        ("eval",),
    ]
    assert events[3][:3] == ("KPipeline", "a", "hexgrad/Kokoro-82M")
    assert isinstance(events[3][3], FakeKModel) and events[3][4] == "cpu"
    assert model.model_name == "kokoro/Kokoro-82M/af_heart"

    audio = model.synthesize("hello", "en")
    assert events[-1] == ("call", "hello", str(tmp_path / "voices" / "af_heart.pt"))
    assert (audio.sample_rate, audio.duration_ms) == (24_000, 100)


def test_kokoro_voice_without_a_pin_is_passed_by_name_and_warns(monkeypatch, caplog, tmp_path):
    events: list = []
    fake_kokoro(monkeypatch, events)
    files = {
        ("hexgrad/Kokoro-82M", "config.json"): tmp_path / "config.json",
        ("hexgrad/Kokoro-82M", "kokoro-v1_0.pth"): tmp_path / "kokoro-v1_0.pth",
    }
    monkeypatch.setattr(tts, "pinned_file", lambda repo, filename: files[(repo, filename)])

    with caplog.at_level(logging.WARNING, logger="app.services.tts"):
        model = KokoroTextToSpeech(voice="af_bella", device="cpu")
    model.synthesize("hello", "en")

    assert events[-1] == ("call", "hello", "af_bella")
    assert model.model_name == "kokoro/Kokoro-82M/af_bella"
    assert "af_bella" in caplog.text

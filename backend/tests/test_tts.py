# [파일 역할] 음성 합성 서비스(app/services/tts.py)에서 모델 없이 시험할 수 있는 공통 도우미를 검증한다: 파형
# → WAV 변환(wav_bytes), 모델 실패를 ModelError로 바꾸는 경계(_guarded), 입력 검사(_check_text), 언어별 모델
# 연결(LanguageTextToSpeech).
# [왜 중요한가] 합성 모델(MeloTTS·Kokoro 등)마다 출력 형식과 실패 방식이 다르다. 이 도우미들이 서비스 쪽에 늘
# 같은 WAV 형식과 같은 오류 종류(ModelError)를 주어야, 합성이 실패해도 번역은 돌려주는 처리(README 설계
# 판단)가 동작한다.

import io
import wave

import numpy as np
import pytest

# _guarded·_check_text는 모듈 내부 도우미지만 동작을 직접 고정하려고 가져온다. FakeTextToSpeech는 단어 수 ×
# 100 ms 길이의 무음을 만드는 가짜 합성기다.
from app.services.interfaces import ModelError
from app.services.tts import LanguageTextToSpeech, _check_text, _guarded, wav_bytes
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
    assert audio.duration_ms == 0  # 5 samples at 16 kHz round to 0 ms


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

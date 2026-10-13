"""Supertonic 3 on ONNX Runtime (T78, docs/experiments.md 12), with fake ONNX sessions: the real model files
are a 400 MB download (eval.supertonic_download) that tests and CI do not have."""

# [파일 역할] 한국어 합성 비교용 설정(KO_TTS=supertonic) Supertonic 3 엔진(app/services/supertonic.py)과
# 서비스 포장(tts.py의 SupertonicTextToSpeech)을 가짜 ONNX 세션으로 검증한다. 실제 모델 파일(약 400MB)은
# 테스트·CI에 없다.
# [무엇을 확인하나] 참조 코드를 옮긴 글자 전처리(NFKD 분해, 기호 정리, 문장 끝 마침표, 언어 꼬리표, 읽을 수
# 없는 글자 버리기), 긴 글 나누기, 추론 단계(길이 예측 → 글자 인코딩 → N단계 flow matching → 보코더)의
# 입력·출력 모양, 고정 SHA-256 파일 확인, CPU·CUDA 제공자 설정과 CUDA에서 CPU로 몰래 넘어가는 경우의 차단.
# [배경] Supertonic 3는 시험(docs/experiments.md 12·14절)에서 채택되지 않아 기본은 MeloTTS다. 만든 곳이
# 저장소를 보관 처리해 파일을 리비전·해시로 고정해 받는다(README).

import hashlib
import io
import json
import unicodedata
import wave

import numpy as np
import pytest

# supertonic 모듈 상수(ONNX_PARTS, PINNED_FILES, DEFAULT_VOICE)와 엔진·전처리·나누기·파일 확인 함수, 그리고
# 서비스가 쓰는 포장 클래스를 가져온다.
from app.services import supertonic
from app.services.interfaces import ModelError
from app.services.supertonic import SupertonicEngine, TextProcessor, chunk_text, verify_files
from app.services.tts import SupertonicTextToSpeech

# 가짜 색인표에서 '읽을 수 없음'(-1)으로 둘 글자. CONFIG는 실제 tts.json의 필요한 항목만 작은 값으로 흉내 낸
# 설정이다(표본화율 1000 Hz라 길이 계산을 손으로 할 수 있다).
UNSUPPORTED = "€"
CONFIG = {
    "ae": {"sample_rate": 1000, "base_chunk_size": 10},
    "ttl": {"latent_dim": 2, "chunk_compress_factor": 3},
}
# 잠재 프레임 하나가 만드는 표본 수(base_chunk_size × chunk_compress_factor = 10 × 3).
CHUNK = 30  # samples per latent frame: base_chunk_size * chunk_compress_factor


# 실제 unicode_indexer.json을 대신하는 색인표: 유니코드 코드값 % 1000을 색인으로 주고, UNSUPPORTED만 -1로
# 둔다.
def indexer() -> list[int]:
    table = [code % 1000 for code in range(65536)]
    table[ord(UNSUPPORTED)] = -1
    return table


# 기대값도 같은 NFKD 분해로 만들려는 도우미(한글 음절이 초성·중성·종성 자모로 나뉜다).
def nfkd(text: str) -> str:
    return unicodedata.normalize("NFKD", text)


# ONNX Runtime InferenceSession의 대역. 받은 입력(feeds)을 기록하고, 정해 준 함수의 결과 하나를 출력 목록으로
# 돌려준다.
class FakeSession:
    def __init__(self, fn) -> None:
        self.fn = fn
        self.calls: list[dict] = []

    def run(self, outputs, feeds):
        self.calls.append(feeds)
        return [self.fn(feeds)]


class ZeroNoise:
    """Stands in for the random generator, so the flow-matching steps can be counted in the output."""

    def standard_normal(self, shape, dtype):
        return np.zeros(shape, dtype=dtype)


# 네 그래프의 가짜 세션 묶음. 길이 예측기는 정해 준 초를, 글자 인코더는 0 배열을, 벡터 추정기는 입력에 1을
# 더한 값을(그래서 단계 수가 출력에 남는다), 보코더는 잠재 평균 ÷ 100으로 채운 파형을 낸다.
def fake_sessions(seconds: float = 0.5) -> dict[str, FakeSession]:
    return {
        "duration_predictor": FakeSession(lambda feeds: np.array([seconds], dtype=np.float32)),
        "text_encoder": FakeSession(
            lambda feeds: np.zeros((1, 256, feeds["text_ids"].shape[1]), dtype=np.float32)
        ),
        # Each step adds 1, so the vocoder sees the number of steps that ran.
        "vector_estimator": FakeSession(lambda feeds: feeds["noisy_latent"] + 1),
        "vocoder": FakeSession(
            lambda feeds: np.full(
                (1, feeds["latent"].shape[2] * CHUNK), feeds["latent"].mean() / 100, np.float32
            )
        ),
    }


# 가짜 세션·가짜 목소리·0 잡음으로 엔진을 만든다. 실제 목소리 파일의 배열 모양(ttl (1, 50, 256), dp (1, 8,
# 16))만 맞춘다.
def engine(sessions=None) -> SupertonicEngine:
    voice = supertonic.Voice(ttl=np.zeros((1, 50, 256), np.float32), dp=np.zeros((1, 8, 16), np.float32))
    return SupertonicEngine(
        CONFIG, TextProcessor(indexer()), sessions or fake_sessions(), voice, rng=ZeroNoise()
    )


# 한국어 문장이 NFKD로 분해되고, 끝에 문장부호가 없으면 마침표가 붙으며, <ko>…</ko> 꼬리표로 감싸이는지
# 확인한다. 이미 물음표로 끝나면 마침표를 더 붙이지 않는다. 두 번째 값은 버린 글자 수(0)다.
def test_korean_text_is_decomposed_ended_and_tagged():
    assert TextProcessor(indexer()).prepare("안녕하세요", "ko") == (f"<ko>{nfkd('안녕하세요.')}</ko>", 0)
    # Text that already ends with punctuation gets no extra period.
    assert TextProcessor(indexer()).prepare("정말요?", "ko") == (f"<ko>{nfkd('정말요?')}</ko>", 0)


# 참조 코드와 같은 기호 정리: 둥근 따옴표는 곧은 따옴표로, en 대시는 하이픈으로, 슬래시는 공백으로 바꾸고
# 공백을 하나로 줄인다.
def test_symbols_are_normalized_as_in_the_reference_code():
    prepared, _ = TextProcessor(indexer()).prepare("“네” – 좋아요 / 정말  ", "ko")
    assert prepared == f"<ko>{nfkd(chr(34) + '네' + chr(34) + ' - 좋아요 정말.')}</ko>"


# 색인이 없는 글자는 공백으로 바꿔 버리고 그 수(2)를 센다. 참조 코드는 이런 글자에 -1을 모델로 넘겼으므로 이
# 프로젝트에서 바꾼 부분이다(모듈 docstring).
def test_characters_the_model_cannot_read_are_dropped_and_counted():
    assert TextProcessor(indexer()).prepare(f"가{UNSUPPORTED}나 {UNSUPPORTED}", "ko") == (
        f"<ko>{nfkd('가 나.')}</ko>",
        2,
    )


# 읽을 수 있는 글자가 하나도 남지 않으면 빈 입력을 모델에 넘기지 않고 ValueError로 거절한다.
def test_text_with_nothing_the_model_can_read_is_refused():
    with pytest.raises(ValueError, match="nothing"):
        TextProcessor(indexer()).prepare(f"{UNSUPPORTED} {UNSUPPORTED}", "ko")


# 꼬리표까지 포함한 글자마다 색인표의 값이 int64 id가 되고, 마스크는 (1, 1, 길이)의 float32 1로 채워지는지
# 확인한다(모델 입력 모양).
def test_ids_follow_the_unicode_indexer():
    ids, mask = TextProcessor(indexer()).ids("<ko>a</ko>")
    assert ids.dtype == np.int64 and ids.shape == (1, 10)
    assert ids[0].tolist() == [ord(char) % 1000 for char in "<ko>a</ko>"]
    assert mask.shape == (1, 1, 10) and mask.dtype == np.float32 and mask.min() == 1.0


# 한국어는 120자 안에서 문장 끝으로 나눈다: 71자 문장 둘은 합치면 넘치므로 둘로, 71자 문장과 짧은 문장은 한
# 덩어리로.
def test_long_korean_text_is_cut_at_sentence_ends_into_120_character_chunks():
    first, second = "가" * 70 + ".", "나" * 70 + "."
    assert chunk_text(f"{first} {second}", 120) == [first, second]
    assert chunk_text(f"{first} 다.", 120) == [f"{first} 다."]


# 덩어리마다 예측 길이(0.5초 ÷ 기본 속도 1.05 → 476표본)로 자르고, 덩어리 사이에 0.3초(300표본) 무음을 넣어
# 잇는지 확인한다. 각 덩어리를 자기 예측 길이로 자르는 것도 참조 코드에서 바꾼 부분이다(모듈 docstring).
# 잡음 잠재의 모양은 ceil(476 / 30) = 16프레임 × (latent_dim 2 × 압축 3 = 6) 채널이다. 2단계를 돌았으니 보코더
# 출력은 2 / 100 = 0.02다.
def test_each_chunk_is_trimmed_to_its_predicted_length_and_joined_by_silence():
    sessions = fake_sessions(seconds=0.5)
    audio = engine(sessions).synthesize("가" * 70 + ". " + "나" * 70 + ".", "ko", steps=2)
    # 0.5 s predicted, divided by the default speed 1.05: int(476.19) samples at 1000 Hz per chunk.
    assert audio.dtype == np.float32
    assert audio.shape == (476 + 300 + 476,)
    assert np.all(audio[476:776] == 0)  # 0.3 s between the chunks
    assert np.allclose(audio[:476], 0.02) and np.allclose(audio[776:], 0.02)  # 2 steps -> latent 2 -> 2/100
    # The noise covers ceil(476 / 30) = 16 latent frames of 2 * 3 channels.
    assert sessions["vector_estimator"].calls[0]["noisy_latent"].shape == (1, 6, 16)


# 벡터 추정기가 정확히 steps번(0~7단계) 불리고, 매번 전체 단계 수(8)를 함께 받는지 확인한다. 길이 예측기·글자
# 인코더가 받는 입력 이름도 모델 그래프의 입력 이름과 같아야 한다.
def test_the_vector_estimator_runs_the_given_number_of_steps():
    sessions = fake_sessions()
    audio = engine(sessions).synthesize("안녕하세요.", "ko", steps=8)
    calls = sessions["vector_estimator"].calls
    assert [call["current_step"].tolist() for call in calls] == [[float(step)] for step in range(8)]
    assert all(call["total_step"].tolist() == [8.0] for call in calls)
    assert np.allclose(audio, 0.08)
    feeds = sessions["duration_predictor"].calls[0]
    assert set(feeds) == {"text_ids", "style_dp", "text_mask"}
    assert set(sessions["text_encoder"].calls[0]) == {"text_ids", "style_ttl", "text_mask"}


# 고정 SHA-256과 같으면 통과, 내용이 다르면 그 파일 이름을 담은 ValueError, 파일이 없으면 내려받기
# 명령(eval.supertonic_download)을 알려 주는 FileNotFoundError인지 확인한다. 보관 처리돼 더 고쳐지지 않는
# 원본이므로 받은 파일이 정확히 그 리비전인지 늘 확인한다.
def test_pinned_files_are_checked_by_sha_256(tmp_path):
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx" / "a.onnx").write_bytes(b"abc")
    good = {"onnx/a.onnx": hashlib.sha256(b"abc").hexdigest()}
    verify_files(tmp_path, good)
    with pytest.raises(ValueError, match="onnx/a.onnx"):
        verify_files(tmp_path, {"onnx/a.onnx": hashlib.sha256(b"abd").hexdigest()})
    with pytest.raises(FileNotFoundError, match="supertonic_download"):
        verify_files(tmp_path, {**good, "onnx/b.onnx": "0" * 64})


# 엔진이 읽는 모든 파일(ONNX 그래프 넷, 설정, 색인표, 기본 목소리)이 고정 목록에 있고, 모든 해시가
# 64자리(SHA-256)인지 확인한다. 고정하지 않은 파일을 읽는 일을 막는다.
def test_the_pin_covers_every_file_the_engine_reads():
    names = {f"onnx/{part}.onnx" for part in supertonic.ONNX_PARTS}
    names |= {"onnx/tts.json", "onnx/unicode_indexer.json", f"voice_styles/{supertonic.DEFAULT_VOICE}.json"}
    assert names <= set(supertonic.PINNED_FILES)
    assert all(len(digest) == 64 for digest in supertonic.PINNED_FILES.values())


# load가 읽는 설정·색인표·목소리 파일만 임시 폴더에 쓴다(ONNX 파일은 가짜 세션이 대신하므로 만들지 않는다).
def write_model_dir(path):
    (path / "onnx").mkdir()
    (path / "voice_styles").mkdir()
    (path / "onnx" / "tts.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    (path / "onnx" / "unicode_indexer.json").write_text(json.dumps(indexer()), encoding="utf-8")
    style = {
        "style_ttl": {"dims": [1, 50, 256], "data": [0.0] * (50 * 256)},
        "style_dp": {"dims": [1, 8, 16], "data": [0.0] * (8 * 16)},
    }
    (path / "voice_styles" / "F1.json").write_text(json.dumps(style), encoding="utf-8")


# onnxruntime의 InferenceSession·get_available_providers·preload_dlls를 monkeypatch로 바꿔, 세션마다 어떤
# 경로·옵션·제공자로 만들어졌는지와 CUDA DLL 미리 불러오기 호출을 기록한다. active를 주면 세션이 실제로 얻은
# 제공자를 다르게 보고하게 할 수 있다(CPU로 넘어간 경우 흉내).
def recording_runtime(monkeypatch, available=("CUDAExecutionProvider", "CPUExecutionProvider"), active=None):
    """Stands in for ONNX Runtime's sessions: records what each session was asked for and reports `active`
    as the providers it got (by default the ones it was asked for)."""
    import onnxruntime

    built, preloaded = [], []

    class RecordingSession:
        def __init__(self, path, sess_options=None, providers=None):
            built.append((path, sess_options, providers))
            asked = [provider[0] if isinstance(provider, tuple) else provider for provider in providers]
            self._active = list(active) if active is not None else asked

        def get_providers(self):
            return self._active

    monkeypatch.setattr(onnxruntime, "InferenceSession", RecordingSession)
    monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: list(available))
    monkeypatch.setattr(
        onnxruntime, "preload_dlls", lambda *args, **kwargs: preloaded.append(kwargs), raising=False
    )
    return built, preloaded


# CPU 설정: 네 그래프 모두 CPUExecutionProvider로 만들고, 연산 안 스레드 수는 주어진 값(3), 연산 사이 스레드는
# 1로 고정하는지 확인한다. 설정 파일의 표본화율을 읽고, 세션별 실제 제공자를 기록하며, CPU에서는 CUDA
# 라이브러리를 불러오지 않는다.
def test_sessions_run_on_the_cpu_with_a_fixed_number_of_threads(tmp_path, monkeypatch):
    write_model_dir(tmp_path)
    built, preloaded = recording_runtime(monkeypatch, available=("CPUExecutionProvider",))
    loaded = SupertonicEngine.load(tmp_path, threads=3, verify=False)
    assert sorted(path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] for path, _, _ in built) == sorted(
        f"{part}.onnx" for part in supertonic.ONNX_PARTS
    )
    assert all(providers == ["CPUExecutionProvider"] for _, _, providers in built)
    assert all(
        options.intra_op_num_threads == 3 and options.inter_op_num_threads == 1 for _, options, _ in built
    )
    assert loaded.sample_rate == 1000
    assert loaded.providers == {part: ["CPUExecutionProvider"] for part in supertonic.ONNX_PARTS}
    assert preloaded == []  # the CPU needs no CUDA libraries


# CUDA 설정: 측정 전에 정한 제공자 옵션(GPU 0번, 합성곱 알고리즘 HEURISTIC 탐색, 요청한 만큼만 메모리 늘리기,
# 메모리 한도 1GB)과 CPU 예비 제공자로 세션을 만들고, 세션을 만들기 전에 pip 패키지의 CUDA·cuDNN 라이브러리를
# 한 번 불러오는지 확인한다. HEURISTIC은 기본값 EXHAUSTIVE가 입력 모양이 바뀔 때마다(문장마다) 다시 탐색하기
# 때문이다(테스트 주석, docs/experiments.md 14절).
def test_cuda_sessions_get_the_provider_options_fixed_before_measuring(tmp_path, monkeypatch):
    # T82 (docs/experiments.md 14): ORT's default EXHAUSTIVE convolution search would search again for every
    # new input shape, and speech changes shape with every sentence.
    write_model_dir(tmp_path)
    built, preloaded = recording_runtime(monkeypatch)
    loaded = SupertonicEngine.load(tmp_path, threads=3, verify=False, provider="cuda")
    options = {
        "device_id": 0,
        "cudnn_conv_algo_search": "HEURISTIC",
        "arena_extend_strategy": "kSameAsRequested",
        "gpu_mem_limit": 1 << 30,
    }
    assert len(built) == len(supertonic.ONNX_PARTS)
    assert all(
        providers == [("CUDAExecutionProvider", options), "CPUExecutionProvider"] for _, _, providers in built
    )
    assert len(preloaded) == 1  # the CUDA and cuDNN libraries of the pip packages, before any session
    assert loaded.providers == {
        part: ["CUDAExecutionProvider", "CPUExecutionProvider"] for part in supertonic.ONNX_PARTS
    }


# ONNX Runtime은 CUDA 제공자를 시작하지 못하면 경고만 하고 CPU로 돌린다. 그러면 'GPU 측정'이 실제로는 CPU
# 측정이 되므로, 세션 하나라도 CUDA가 아니면 적재를 RuntimeError로 멈추는지 확인한다.
def test_a_session_that_fell_back_to_the_cpu_stops_the_cuda_load(tmp_path, monkeypatch):
    # ONNX Runtime only warns when the CUDA provider cannot start and runs the graph on the CPU instead.
    write_model_dir(tmp_path)
    recording_runtime(monkeypatch, active=["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
        SupertonicEngine.load(tmp_path, verify=False, provider="cuda")


# 설치된 onnxruntime에 CUDA 제공자가 없으면(CPU 전용 패키지) onnxruntime-gpu가 필요하다는 오류로 멈추고,
# 세션은 하나도 만들지 않는지 확인한다.
def test_the_cuda_provider_needs_the_gpu_build_of_onnxruntime(tmp_path, monkeypatch):
    write_model_dir(tmp_path)
    built, _ = recording_runtime(monkeypatch, available=("AzureExecutionProvider", "CPUExecutionProvider"))
    with pytest.raises(RuntimeError, match="onnxruntime-gpu"):
        SupertonicEngine.load(tmp_path, verify=False, provider="cuda")
    assert built == []


# cpu·cuda 외의 제공자 이름은 거절한다.
def test_an_unknown_provider_is_refused(tmp_path):
    with pytest.raises(ValueError, match="provider"):
        SupertonicEngine.load(tmp_path, verify=False, provider="tensorrt")


# 고정 목록에 없는 목소리 파일은 파일 확인 전에 거절한다.
def test_a_voice_outside_the_pin_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not pinned"):
        SupertonicEngine.load(tmp_path, voice="Z9")


# 서비스 포장 확인: 모델 이름은 '모델/목소리/단계 수' 형식이고(/api/health와 기록에 남음), 결과는 16비트 모노
# WAV(표본화율 1000, 476 ms)다. 한국어만 맡으므로 영어 요청과, 엔진 안의 예외(보코더 RuntimeError), 읽을 수
# 없는 글자뿐인 입력은 모두 ModelError가 된다. 그래야 서비스가 '합성 실패, 번역은 돌려줌'으로 처리할 수 있다.
def test_the_service_wrapper_speaks_korean_as_a_wav_and_reports_failures_as_model_errors():
    tts = SupertonicTextToSpeech(steps=2, engine=engine())
    assert tts.model_name == "supertonic-3/F1/2-step"
    audio = tts.synthesize("안녕하세요.", "ko")
    with wave.open(io.BytesIO(audio.wav)) as clip:
        assert (clip.getnchannels(), clip.getsampwidth(), clip.getframerate()) == (1, 2, 1000)
    assert audio.sample_rate == 1000 and audio.duration_ms == 476
    with pytest.raises(ModelError, match="does not speak en"):
        tts.synthesize("hello", "en")

    broken = fake_sessions()
    broken["vocoder"] = FakeSession(lambda feeds: (_ for _ in ()).throw(RuntimeError("bad shape")))
    with pytest.raises(ModelError, match="RuntimeError"):
        SupertonicTextToSpeech(steps=2, engine=engine(broken)).synthesize("안녕하세요.", "ko")
    with pytest.raises(ModelError):
        SupertonicTextToSpeech(steps=2, engine=engine()).synthesize(UNSUPPORTED, "ko")


# CUDA 설정이면 모델 이름 끝에 /cuda가 붙고 세션별 실제 제공자를 보여 주며, CPU 설정의 이름은 이전
# 시험(12절)에 기록된 이름 그대로인지 확인한다.
def test_the_service_wrapper_names_the_gpu_and_reports_the_session_providers(tmp_path, monkeypatch):
    write_model_dir(tmp_path)
    recording_runtime(monkeypatch)
    on_gpu = SupertonicTextToSpeech(tmp_path, steps=2, verify=False, provider="cuda")
    assert on_gpu.model_name == "supertonic-3/F1/2-step/cuda"
    assert on_gpu.providers["vocoder"][0] == "CUDAExecutionProvider"
    # The CPU name stays what T78 recorded.
    assert SupertonicTextToSpeech(tmp_path, steps=2, verify=False).model_name == "supertonic-3/F1/2-step"

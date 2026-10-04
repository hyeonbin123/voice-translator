"""Supertonic 3 on ONNX Runtime (T78, docs/experiments.md 12), with fake ONNX sessions: the real model files
are a 400 MB download (eval.supertonic_download) that tests and CI do not have."""

import hashlib
import io
import json
import unicodedata
import wave

import numpy as np
import pytest

from app.services import supertonic
from app.services.interfaces import ModelError
from app.services.supertonic import SupertonicEngine, TextProcessor, chunk_text, verify_files
from app.services.tts import SupertonicTextToSpeech

UNSUPPORTED = "€"
CONFIG = {
    "ae": {"sample_rate": 1000, "base_chunk_size": 10},
    "ttl": {"latent_dim": 2, "chunk_compress_factor": 3},
}
CHUNK = 30  # samples per latent frame: base_chunk_size * chunk_compress_factor


def indexer() -> list[int]:
    table = [code % 1000 for code in range(65536)]
    table[ord(UNSUPPORTED)] = -1
    return table


def nfkd(text: str) -> str:
    return unicodedata.normalize("NFKD", text)


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


def engine(sessions=None) -> SupertonicEngine:
    voice = supertonic.Voice(ttl=np.zeros((1, 50, 256), np.float32), dp=np.zeros((1, 8, 16), np.float32))
    return SupertonicEngine(
        CONFIG, TextProcessor(indexer()), sessions or fake_sessions(), voice, rng=ZeroNoise()
    )


def test_korean_text_is_decomposed_ended_and_tagged():
    assert TextProcessor(indexer()).prepare("안녕하세요", "ko") == (f"<ko>{nfkd('안녕하세요.')}</ko>", 0)
    # Text that already ends with punctuation gets no extra period.
    assert TextProcessor(indexer()).prepare("정말요?", "ko") == (f"<ko>{nfkd('정말요?')}</ko>", 0)


def test_symbols_are_normalized_as_in_the_reference_code():
    prepared, _ = TextProcessor(indexer()).prepare("“네” – 좋아요 / 정말  ", "ko")
    assert prepared == f"<ko>{nfkd(chr(34) + '네' + chr(34) + ' - 좋아요 정말.')}</ko>"


def test_characters_the_model_cannot_read_are_dropped_and_counted():
    assert TextProcessor(indexer()).prepare(f"가{UNSUPPORTED}나 {UNSUPPORTED}", "ko") == (
        f"<ko>{nfkd('가 나.')}</ko>",
        2,
    )


def test_text_with_nothing_the_model_can_read_is_refused():
    with pytest.raises(ValueError, match="nothing"):
        TextProcessor(indexer()).prepare(f"{UNSUPPORTED} {UNSUPPORTED}", "ko")


def test_ids_follow_the_unicode_indexer():
    ids, mask = TextProcessor(indexer()).ids("<ko>a</ko>")
    assert ids.dtype == np.int64 and ids.shape == (1, 10)
    assert ids[0].tolist() == [ord(char) % 1000 for char in "<ko>a</ko>"]
    assert mask.shape == (1, 1, 10) and mask.dtype == np.float32 and mask.min() == 1.0


def test_long_korean_text_is_cut_at_sentence_ends_into_120_character_chunks():
    first, second = "가" * 70 + ".", "나" * 70 + "."
    assert chunk_text(f"{first} {second}", 120) == [first, second]
    assert chunk_text(f"{first} 다.", 120) == [f"{first} 다."]


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


def test_pinned_files_are_checked_by_sha_256(tmp_path):
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx" / "a.onnx").write_bytes(b"abc")
    good = {"onnx/a.onnx": hashlib.sha256(b"abc").hexdigest()}
    verify_files(tmp_path, good)
    with pytest.raises(ValueError, match="onnx/a.onnx"):
        verify_files(tmp_path, {"onnx/a.onnx": hashlib.sha256(b"abd").hexdigest()})
    with pytest.raises(FileNotFoundError, match="supertonic_download"):
        verify_files(tmp_path, {**good, "onnx/b.onnx": "0" * 64})


def test_the_pin_covers_every_file_the_engine_reads():
    names = {f"onnx/{part}.onnx" for part in supertonic.ONNX_PARTS}
    names |= {"onnx/tts.json", "onnx/unicode_indexer.json", f"voice_styles/{supertonic.DEFAULT_VOICE}.json"}
    assert names <= set(supertonic.PINNED_FILES)
    assert all(len(digest) == 64 for digest in supertonic.PINNED_FILES.values())


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


def test_a_session_that_fell_back_to_the_cpu_stops_the_cuda_load(tmp_path, monkeypatch):
    # ONNX Runtime only warns when the CUDA provider cannot start and runs the graph on the CPU instead.
    write_model_dir(tmp_path)
    recording_runtime(monkeypatch, active=["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
        SupertonicEngine.load(tmp_path, verify=False, provider="cuda")


def test_the_cuda_provider_needs_the_gpu_build_of_onnxruntime(tmp_path, monkeypatch):
    write_model_dir(tmp_path)
    built, _ = recording_runtime(monkeypatch, available=("AzureExecutionProvider", "CPUExecutionProvider"))
    with pytest.raises(RuntimeError, match="onnxruntime-gpu"):
        SupertonicEngine.load(tmp_path, verify=False, provider="cuda")
    assert built == []


def test_an_unknown_provider_is_refused(tmp_path):
    with pytest.raises(ValueError, match="provider"):
        SupertonicEngine.load(tmp_path, verify=False, provider="tensorrt")


def test_a_voice_outside_the_pin_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not pinned"):
        SupertonicEngine.load(tmp_path, voice="Z9")


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


def test_the_service_wrapper_names_the_gpu_and_reports_the_session_providers(tmp_path, monkeypatch):
    write_model_dir(tmp_path)
    recording_runtime(monkeypatch)
    on_gpu = SupertonicTextToSpeech(tmp_path, steps=2, verify=False, provider="cuda")
    assert on_gpu.model_name == "supertonic-3/F1/2-step/cuda"
    assert on_gpu.providers["vocoder"][0] == "CUDAExecutionProvider"
    # The CPU name stays what T78 recorded.
    assert SupertonicTextToSpeech(tmp_path, steps=2, verify=False).model_name == "supertonic-3/F1/2-step"

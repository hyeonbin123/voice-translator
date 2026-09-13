import io
import wave
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import av
import pytest
from sqlalchemy import func, select

from app.core.security import create_token
from app.dependencies import get_audio_store, get_models
from app.main import app
from app.models import AudioFile, Translation, User
from app.services import stt, translation, tts
from app.services.audio_store import AudioStore
from app.services.interfaces import ModelError, NoSpeechError, UndecodableAudioError
from app.services.pipeline import MAX_AUDIO_BYTES, PipelineModels
from tests.fakes import FakeSpeechToText, FakeTextToSpeech, FakeTranslator, silent_wav


@pytest.fixture
def services(client, tmp_path):
    models = PipelineModels(FakeSpeechToText(text="recognized words"), FakeTranslator(), FakeTextToSpeech())
    store = AudioStore(tmp_path / "audio")
    app.dependency_overrides[get_models] = lambda: models
    app.dependency_overrides[get_audio_store] = lambda: store
    return models, store


async def text_request(client, headers, **changes):
    body = {"text": "  안녕하세요  ", "source_lang": "ko", "target_lang": "en", **changes}
    return await client.post("/api/translate/text", headers=headers, json=body)


async def speech_request(client, headers, content=None, **changes):
    return await client.post(
        "/api/translate/speech",
        headers=headers,
        files={
            "audio": (
                "untrusted.exe",
                silent_wav(100) if content is None else content,
                "application/octet-stream",
            )
        },
        data={"source_lang": "en", "target_lang": "ko", **changes},
    )


async def assert_no_history(client, headers, db):
    assert (await client.get("/api/history", headers=headers)).json() == {"items": [], "total": 0}
    assert await db.scalar(select(func.count()).select_from(AudioFile)) == 0


@pytest.mark.parametrize("speech", [False, True])
async def test_translation_round_trip(client, auth_headers, services, db_session, speech):
    _, store = services
    response = await (speech_request(client, auth_headers) if speech else text_request(client, auth_headers))
    assert response.status_code == 201, response.text
    result = response.json()
    assert response.headers["location"] == f"/api/history/{result['id']}"
    assert result["mode"] == ("speech" if speech else "text")
    assert result["source_text"] == ("recognized words" if speech else "안녕하세요")
    assert result["translated_text"] == ("[en->ko] recognized words" if speech else "[ko->en] 안녕하세요")
    assert result["source_lang"] == ("en" if speech else "ko")
    assert result["target_lang"] == ("ko" if speech else "en")
    assert result["stt_model"] == ("fake-stt" if speech else None)
    assert (
        (isinstance(result["stt_ms"], int) and result["stt_ms"] >= 0)
        if speech
        else (result["stt_ms"] is None)
    )
    assert result["mt_model"] == "fake-mt" and result["tts_model"] == "fake-tts"
    assert result["mt_ms"] >= 0 and result["tts_ms"] >= 0 and result["tts_error"] is None
    history = (await client.get(response.headers["location"], headers=auth_headers)).json()
    assert history == {key: value for key, value in result.items() if key != "tts_error"}
    audio = await client.get(f"/api/audio/{result['audio_id']}", headers=auth_headers)
    assert audio.status_code == 200 and audio.headers["content-type"] == "audio/wav"
    assert audio.headers["cache-control"] == "private, no-store"
    with wave.open(io.BytesIO(audio.content)) as wav:
        assert wav.getsampwidth() == 2 and wav.getnchannels() == 1 and wav.getnframes() > 0
    row = await db_session.get(AudioFile, UUID(result["audio_id"]))
    assert store.resolve(row.path).read_bytes() == audio.content
    assert len(list(store.directory.iterdir())) == 1  # Only synthesis; no original upload.
    assert "path" not in result and "user_id" not in result


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"text": ""},
        {"text": " \n\t "},
        {"text": "가" * 501},
        {"text": 123},
        {"source_lang": "fr"},
        {"source_lang": "en"},
        {"target_lang": None},
    ],
)
async def test_invalid_text(client, auth_headers, services, db_session, body):
    response = (
        await client.post("/api/translate/text", headers=auth_headers, json={})
        if not body
        else await text_request(client, auth_headers, **body)
    )
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)
    await assert_no_history(client, auth_headers, db_session)


async def test_unicode_500_after_trim(client, auth_headers, services):
    response = await text_request(client, auth_headers, text=" \n" + "😀" * 500 + " \t")
    assert response.status_code == 201
    assert response.json()["source_text"] == "😀" * 500


@pytest.mark.parametrize("changes", [{"source_lang": "fr"}, {"target_lang": "en"}])
async def test_invalid_speech_languages(client, auth_headers, services, db_session, changes):
    response = await speech_request(client, auth_headers, **changes)
    assert response.status_code == 422 and isinstance(response.json()["detail"], list)
    await assert_no_history(client, auth_headers, db_session)


@pytest.mark.parametrize("field", ["audio", "source_lang", "target_lang"])
async def test_missing_speech_fields(client, auth_headers, services, field):
    files = {"audio": ("test.wav", silent_wav(100))} if field != "audio" else {}
    data = {key: value for key, value in {"source_lang": "en", "target_lang": "ko"}.items() if key != field}
    response = await client.post("/api/translate/speech", headers=auth_headers, files=files, data=data)
    assert response.status_code == 422 and isinstance(response.json()["detail"], list)


@pytest.mark.parametrize(
    "content,detail",
    [
        (b"", "Audio could not be decoded"),
        (b"not audio", "Audio could not be decoded"),
        (silent_wav(30001), "Audio is longer than 30 seconds"),
    ],
    ids=["empty", "garbage", "over-30-seconds"],
)
async def test_invalid_audio_before_stt(client, auth_headers, services, db_session, content, detail):
    models, _ = services
    # If validation calls STT, this would become 503 instead of the expected 422.
    models.stt.error = ModelError("STT must not execute")
    response = await speech_request(client, auth_headers, content)
    assert response.status_code == 422 and response.json() == {"detail": detail}
    await assert_no_history(client, auth_headers, db_session)


async def test_exactly_30_seconds_accepted(client, auth_headers, services):
    response = await speech_request(client, auth_headers, silent_wav(30000))
    assert response.status_code == 201, response.text


@pytest.mark.parametrize(
    "component,error,status,detail",
    [
        ("stt", UndecodableAudioError("private decoder path"), 422, "Audio could not be decoded"),
        ("stt", NoSpeechError("private speech detail"), 422, "No speech was recognized"),
        ("stt", ModelError("private GPU error"), 503, "Translation service is unavailable"),
        ("translator", ModelError("private model path"), 503, "Translation service is unavailable"),
    ],
)
async def test_model_failures(
    client, auth_headers, services, db_session, caplog, component, error, status, detail
):
    models, store = services
    getattr(models, component).error = error
    for _ in range(2):  # Fake failures must repeat, not be consumed once.
        response = await speech_request(client, auth_headers)
        assert response.status_code == status and response.json() == {"detail": detail}
    # Logged by type, never by message (T49): a library's message can quote the input.
    assert type(error).__name__ in caplog.text and str(error) not in caplog.text
    await assert_no_history(client, auth_headers, db_session)
    assert not store.directory.exists()


SOURCE_MARK, TRANSLATION_MARK = "원문표식", "translation-mark"


class FailingWhisper:
    """faster-whisper failing as CTranslate2 does: on the call, or while the segments are read (T48)."""

    lazy = False

    def __init__(self, *args, **kwargs):
        pass

    def transcribe(self, audio, language, **options):
        if not self.lazy:
            raise RuntimeError(f"CUDA failed while decoding {SOURCE_MARK}")

        def segments():
            yield SimpleNamespace(text="partial")
            raise RuntimeError(f"CUDA failed while decoding {SOURCE_MARK}")

        return segments(), SimpleNamespace(duration=1.0)


@pytest.mark.parametrize("lazy", [False, True], ids=["on-the-call", "while-reading"])
async def test_a_recognition_engine_error_answers_503_without_quoting_it(
    client, auth_headers, services, db_session, monkeypatch, caplog, lazy
):
    monkeypatch.setattr(stt, "WhisperModel", FailingWhisper)
    monkeypatch.setattr(stt, "add_cuda_dll_dirs", lambda: None)
    monkeypatch.setattr(FailingWhisper, "lazy", lazy)
    models, _ = services
    whisper = stt.WhisperSpeechToText("tiny", device="cpu", compute_type="int8")
    app.dependency_overrides[get_models] = lambda: replace(models, stt=whisper)
    response = await speech_request(client, auth_headers)
    assert response.status_code == 503 and response.json() == {"detail": "Translation service is unavailable"}
    assert "Translation pipeline failed: ModelError <- RuntimeError" in caplog.text
    assert SOURCE_MARK not in caplog.text and SOURCE_MARK not in response.text
    await assert_no_history(client, auth_headers, db_session)


class QuotingTokenizer:
    """A tokenizer whose error quotes the text, raised while handling another error that quotes it too."""

    def encode(self, text, out_type):
        try:
            raise KeyError(f"no piece for {text}")
        except KeyError:
            raise ValueError(f"cannot encode {text}")  # noqa: B904 - the implicit context is the point

    def decode(self, pieces):
        return " ".join(pieces)


async def test_a_translation_error_quoting_the_input_is_logged_without_it(
    client, auth_headers, services, db_session, monkeypatch, caplog
):
    # The real Marian adapter and its model boundary; only the tokenizer and engine are replaced (T49).
    engine = SimpleNamespace(generate=lambda tokens, target_prefix=None: ["unused"])
    monkeypatch.setattr(translation, "_Ct2Model", lambda *args: engine)
    monkeypatch.setattr(translation, "load_sentencepiece", lambda path: QuotingTokenizer())
    models, _ = services
    marian = translation.MarianTranslator(Path("model"), "ko", "en")
    app.dependency_overrides[get_models] = lambda: replace(models, translator=marian)
    response = await text_request(client, auth_headers, text=f"{SOURCE_MARK} 문장입니다")
    assert response.status_code == 503 and response.json() == {"detail": "Translation service is unavailable"}
    # The stage and the chain of types stay visible; neither message is.
    assert "Translation pipeline failed: ModelError <- ValueError <- KeyError" in caplog.text
    assert SOURCE_MARK not in caplog.text and SOURCE_MARK not in response.text
    await assert_no_history(client, auth_headers, db_session)


class MarkingTranslator:
    model_name = "marking-mt"

    def translate(self, text, source, target):
        return f"{TRANSLATION_MARK} of the text"


class QuotingSynthesis:
    """Speech synthesis whose error, inside the real model boundary, quotes the translation."""

    model_name = "quoting-tts"

    def synthesize(self, text, language):
        def front_end():
            raise ValueError(f"cannot read {text}")

        return tts._guarded(front_end)


async def test_a_synthesis_error_quoting_the_translation_is_logged_without_it(
    client, auth_headers, services, caplog
):
    models, _ = services
    app.dependency_overrides[get_models] = lambda: replace(
        models, translator=MarkingTranslator(), tts=QuotingSynthesis()
    )
    response = await text_request(client, auth_headers, text=f"{SOURCE_MARK} 문장입니다")
    # Synthesis failing still keeps the translation and its record.
    assert response.status_code == 201 and response.json()["tts_error"] == "Speech synthesis failed"
    assert "Speech synthesis or audio storage failed: ModelError <- ValueError" in caplog.text
    assert TRANSLATION_MARK not in caplog.text and SOURCE_MARK not in caplog.text


async def test_fake_empty_recognition(client, auth_headers, services, db_session):
    models, _ = services
    models.stt.text = ""
    response = await speech_request(client, auth_headers)
    assert response.status_code == 422 and response.json() == {"detail": "No speech was recognized"}
    await assert_no_history(client, auth_headers, db_session)


@pytest.mark.parametrize("failure", ["tts", "unavailable", "disk"])
async def test_synthesis_fallback_keeps_history(
    client, auth_headers, services, db_session, monkeypatch, caplog, failure
):
    models, store = services
    if failure == "tts":
        models.tts.fail = True
    elif failure == "unavailable":
        app.dependency_overrides[get_models] = lambda: replace(models, tts=None)
    else:

        async def fail_save(*args):
            raise OSError("private disk error")

        monkeypatch.setattr(store, "save", fail_save)
    response = await text_request(client, auth_headers)
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["tts_error"] == (
        "Speech synthesis is not available" if failure == "unavailable" else "Speech synthesis failed"
    )
    assert result["audio_id"] is None and result["tts_model"] is None and result["tts_ms"] is None
    assert (await client.get("/api/history", headers=auth_headers)).json()["total"] == 1
    assert await db_session.scalar(select(func.count()).select_from(AudioFile)) == 0
    if failure != "unavailable":
        assert "failed" in caplog.text


async def test_missing_required_models(client, auth_headers, services, db_session):
    app.dependency_overrides.pop(get_models)
    response = await text_request(client, auth_headers)
    assert response.status_code == 503
    assert response.json() == {"detail": "Translation service is unavailable"}
    await assert_no_history(client, auth_headers, db_session)


@pytest.mark.parametrize(
    "method,path",
    [
        ("post", "/api/translate/text"),
        ("post", "/api/translate/speech"),
        ("get", f"/api/audio/{uuid4()}"),
    ],
)
async def test_authentication_required(client, method, path):
    response = await getattr(client, method)(path)
    assert response.status_code == 401 and response.headers["www-authenticate"] == "Bearer"


async def test_audio_ownership_and_deletion(client, auth_headers, services, db_session, password_hash):
    _, store = services
    result = (await text_request(client, auth_headers)).json()
    audio_url = f"/api/audio/{result['audio_id']}"
    other = User(email="another@example.com", hashed_password=password_hash)
    db_session.add(other)
    await db_session.commit()
    foreign_headers = {"Authorization": f"Bearer {create_token(other.id, 'access')}"}
    for url, headers in [(audio_url, foreign_headers), (f"/api/audio/{uuid4()}", auth_headers)]:
        response = await client.get(url, headers=headers)
        assert response.status_code == 404 and response.json() == {"detail": "Audio not found"}
    response = await client.delete(f"/api/history/{result['id']}", headers=foreign_headers)
    assert response.status_code == 404
    assert (await client.get(audio_url, headers=auth_headers)).status_code == 200
    response = await client.delete(f"/api/history/{result['id']}", headers=auth_headers)
    assert response.status_code == 204
    assert not list(store.directory.iterdir())
    response = await client.get(audio_url, headers=auth_headers)
    assert response.status_code == 404 and response.json() == {"detail": "Audio not found"}
    assert await db_session.get(AudioFile, UUID(result["audio_id"])) is None
    assert await db_session.get(Translation, UUID(result["id"])) is None
    assert (await client.get("/api/audio/not-a-uuid", headers=auth_headers)).status_code == 422


async def test_missing_disk_file_is_same_404(client, auth_headers, services):
    _, store = services
    result = (await text_request(client, auth_headers)).json()
    next(store.directory.iterdir()).unlink()
    response = await client.get(f"/api/audio/{result['audio_id']}", headers=auth_headers)
    assert response.status_code == 404 and response.json() == {"detail": "Audio not found"}


async def test_disk_delete_failure_keeps_db_deletion(
    client, auth_headers, services, db_session, monkeypatch, caplog
):
    _, store = services
    result = (await text_request(client, auth_headers)).json()

    def fail_unlink(*args, **kwargs):
        raise PermissionError("file in use")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail_unlink)
        response = await client.delete(f"/api/history/{result['id']}", headers=auth_headers)
    assert response.status_code == 204 and "Could not delete audio file" in caplog.text
    assert await db_session.get(Translation, UUID(result["id"])) is None
    assert await db_session.get(AudioFile, UUID(result["audio_id"])) is None
    assert len(list(store.directory.iterdir())) == 1
    assert (await client.get(f"/api/audio/{result['audio_id']}", headers=auth_headers)).status_code == 404


@pytest.mark.parametrize("oversized", [False, True])
async def test_upload_limit_counts_file_bytes(client, auth_headers, services, oversized):
    # A valid WAV padded with trailing bytes verifies exact file bytes, excluding multipart overhead.
    wav = silent_wav(100)
    content = wav + b"\0" * (MAX_AUDIO_BYTES + int(oversized) - len(wav))
    response = await speech_request(client, auth_headers, content)
    assert response.status_code == (413 if oversized else 201), response.text
    if oversized:
        assert response.json() == {"detail": "Audio file is larger than 10 MB"}


async def test_chunked_upload_stops_before_end(client, auth_headers, services, db_session):
    consumed_tail = False

    async def body():
        nonlocal consumed_tail
        yield (
            b'--boundary\r\nContent-Disposition: form-data; name="audio"; filename="x.wav"'
            b"\r\nContent-Type: audio/wav\r\n\r\n"
        )
        for _ in range(MAX_AUDIO_BYTES // 65536 + 1):
            yield b"x" * 65536
        consumed_tail = True
        yield b"\r\n--boundary--\r\n"

    response = await client.post(
        "/api/translate/speech",
        content=body(),
        headers={**auth_headers, "Content-Type": "multipart/form-data; boundary=boundary"},
    )
    assert response.status_code == 413 and not consumed_tail
    await assert_no_history(client, auth_headers, db_session)


async def test_real_webm_decode(client, auth_headers, services):
    output = io.BytesIO()
    with av.open(io.BytesIO(silent_wav(1000))) as source, av.open(output, "w", format="webm") as dest:
        stream = dest.add_stream("libopus", rate=48000)
        for frame in source.decode(audio=0):
            for packet in stream.encode(frame):
                dest.mux(packet)
        for packet in stream.encode(None):
            dest.mux(packet)
    response = await speech_request(client, auth_headers, output.getvalue())
    assert response.status_code == 201, response.text


@pytest.mark.parametrize("content_type", ["multipart/form-data", "multipart/form-data; boundary=abc"])
async def test_malformed_multipart_returns_400(client, auth_headers, services, content_type):
    response = await client.post(
        "/api/translate/speech",
        content=b"not multipart",
        headers={**auth_headers, "Content-Type": content_type},
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid multipart request"}

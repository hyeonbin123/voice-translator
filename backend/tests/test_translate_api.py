# [파일 역할] 글자 번역(POST /api/translate/text)·음성 번역(POST /api/translate/speech)·번역 음성 받기(GET
# /api/audio/{id}) API의 계약을 실제 테스트 DB와 가짜 모델로 검증한다(docs/api.md의 번역 절과 '실패했을 때'
# 표).
# [무엇을 확인하나] 성공 응답·기록·음성 파일이 서로 맞는지, 입력 검증(빈 글자·보이지 않는
# 글자·NUL·500자·언어), 음성 검사(디코딩·30초·10MB), 모델 실패의 고정된 422·503 문구와 기록 없음, 로그에
# 원문·번역문이 남지 않음, 합성 실패 시 번역은 저장, 음성 소유권과 삭제, 업로드 크기 제한과 잘못된 multipart.
# [왜 중요한가] 화면(frontend)은 이 응답 코드와 문구에 맞춰 동작하고, 기록·음성은 사용자별 개인 데이터다.
# 실패한 요청이 기록이나 파일을 남기지 않는 것도 여기서 지킨다.

import io
import wave
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

# av(PyAV): 실제 webm/opus 녹음을 만들어 브라우저 녹음 형식이 디코딩되는지 시험한다. ValidationError:
# TextRequest 규칙을 API 없이 직접 확인할 때 쓴다.
import av
import pytest
from pydantic import ValidationError
from sqlalchemy import func, select

# create_token: 다른 사용자의 토큰을 만들어 남의 음성 접근을 시험한다. stt·translation·tts 모듈은 실제
# 어댑터의 예외 경계를 거치게 하려고 일부만 가짜로 바꿀 때 쓴다. MAX_AUDIO_BYTES는 업로드 한도(10MB)다.
from app.core.security import create_token
from app.dependencies import get_audio_store, get_models
from app.main import app
from app.models import AudioFile, Translation, User
from app.schemas.translate import TextRequest
from app.services import stt, translation, tts
from app.services.audio_store import AudioStore
from app.services.interfaces import ModelError, NoSpeechError, UndecodableAudioError
from app.services.pipeline import MAX_AUDIO_BYTES, PipelineModels
from tests.fakes import FakeSpeechToText, FakeTextToSpeech, FakeTranslator, silent_wav


# 가짜 인식·번역·합성 모델과 임시 폴더 음성 저장소를 앱에 끼우고, 테스트가 실패를 정할 수 있도록 둘을
# 돌려준다.
@pytest.fixture
def services(client, tmp_path):
    models = PipelineModels(FakeSpeechToText(text="recognized words"), FakeTranslator(), FakeTextToSpeech())
    store = AudioStore(tmp_path / "audio")
    app.dependency_overrides[get_models] = lambda: models
    app.dependency_overrides[get_audio_store] = lambda: store
    return models, store


# 글자 번역 요청 도우미. 기본 본문은 앞뒤 공백이 있는 한국어 → 영어이고, changes로 필드를 바꾸거나 더한다.
async def text_request(client, headers, **changes):
    body = {"text": "  안녕하세요  ", "source_lang": "ko", "target_lang": "en", **changes}
    return await client.post("/api/translate/text", headers=headers, json=body)


# 음성 번역 요청 도우미. 파일 이름과 MIME 형식을 일부러 믿을 수 없는 값(untrusted.exe,
# application/octet-stream)으로 보내, 서버가 확장자·형식 대신 실제 내용을 디코딩해 판단하는지 함께 시험한다.
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


# 실패한 요청이 번역 기록도, 음성 행도 남기지 않았는지 확인하는 도우미.
async def assert_no_history(client, headers, db):
    assert (await client.get("/api/history", headers=headers)).json() == {"items": [], "total": 0}
    assert await db.scalar(select(func.count()).select_from(AudioFile)) == 0


# 글자·음성 두 경로의 성공 왕복: 201과 Location 헤더, 모드·원문(글자는 앞뒤 공백을 자른 값)·번역문·방향·모델
# 이름·처리 시간, 기록 상세가 응답과 같은지(tts_error만 응답 전용), 번역 음성이 인증된 요청으로 private,
# no-store 캐시 설정과 함께 내려오는 16비트 모노 WAV이고 디스크의 파일과 같은지 확인한다.
# 저장소에 파일이 정확히 하나(합성 음성)만 있는지도 본다. 녹음 원본은 저장하지 않는다는 설계(README)를
# 지키는지 확인하는 것이다. 응답에 저장 경로·user_id 같은 내부 값도 없어야 한다.
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


# 글자 입력 검증 사례(모두 422, FastAPI 검증 오류 목록 형식, 기록 없음): 빈 본문·빈 글자·공백만, 보이지 않는
# 서식 문자만, 사용자 정의·미배정 문자만, NUL 포함, 501자, 문자열이 아닌 값, 지원하지 않는 언어, 같은
# 언어끼리, 목표 언어 없음. 규칙은 docs/api.md의 text 항목과 같다.
@pytest.mark.parametrize(
    "body",
    [
        {},
        {"text": ""},
        {"text": " \n\t "},
        # Only invisible format characters (zero-width space; BOM and word joiner): nothing to translate.
        {"text": "\u200b"},
        {"text": " \ufeff\u2060 "},
        # Private-use and unassigned characters are "other" (C*) characters too: the model would get spaces.
        {"text": "\ue000\u0378"},
        # PostgreSQL text cannot hold NUL: without the check it failed at the save, after the model work.
        {"text": "a\u0000b"},
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


# 문장부호만 있거나, 화면에 보이지 않아도 모델에 그대로 가는 결합 악센트(U+0301)·이체 선택자(U+FE0F)만 있는
# 입력은 거절하지 않고 짧은 글처럼 번역한다(docs/api.md).
@pytest.mark.parametrize("text", ["?", "́", "️"])
async def test_text_the_model_still_receives_is_translated(client, auth_headers, services, text):
    # Punctuation, or a lone combining accent or variation selector: invisible or not, the model receives
    # it, so it is translated like any short text (docs/api.md).
    response = await text_request(client, auth_headers, text=text)
    assert response.status_code == 201
    assert response.json()["translated_text"] == f"[ko->en] {text}"


# 422 규칙이 실제 번역 앞처리와 정확히 맞는지 확인한다: OPUS-MT 앞처리(줄바꿈을 뺀 C 범주 문자를 공백으로
# 바꿈) 뒤에 공백만 남는 입력일 때, 그리고 그때만 TextRequest가 거절해야 한다. 규칙을 문서·코드와 일치시킨
# 수정(커밋 2800f43, ac6585a)을 사례로 고정한다.
@pytest.mark.parametrize(
    "text",
    ["​", " ﻿⁠ ", "­", "", "͸", "​\n​", "?", "́", "️", "a​b"],
)
def test_the_text_rule_matches_what_the_preprocessing_leaves(text):
    # 422 exactly when OPUS-MT's preprocessing (a space for every C* character but a newline) leaves only
    # white space, that is nothing to translate.
    blank = not translation.opus_preprocess(text.strip()).strip()
    try:
        TextRequest(text=text, source_lang="ko", target_lang="en")
    except ValidationError:
        assert blank
    else:
        assert not blank


# 500자 제한은 앞뒤 공백을 자른 뒤 유니코드 문자(코드 포인트) 수로 센다. 이모지 500개(UTF-16으로는 1000단위)도
# 통과하고, 저장되는 원문은 자른 값이다.
async def test_unicode_500_after_trim(client, auth_headers, services):
    response = await text_request(client, auth_headers, text=" \n" + "😀" * 500 + " \t")
    assert response.status_code == 201
    assert response.json()["source_text"] == "😀" * 500


# 음성 번역의 언어 폼 필드도 글자 번역과 같은 규칙(지원 언어, 서로 다른 언어)으로 422 목록 형식을 내는지
# 확인한다. 폼 필드라 라우터가 LanguagePair를 직접 만들어 검증한다.
@pytest.mark.parametrize("changes", [{"source_lang": "fr"}, {"target_lang": "en"}])
async def test_invalid_speech_languages(client, auth_headers, services, db_session, changes):
    response = await speech_request(client, auth_headers, **changes)
    assert response.status_code == 422 and isinstance(response.json()["detail"], list)
    await assert_no_history(client, auth_headers, db_session)


# 음성 번역의 세 필드(파일, 원문 언어, 목표 언어) 중 하나라도 빠지면 422다.
@pytest.mark.parametrize("field", ["audio", "source_lang", "target_lang"])
async def test_missing_speech_fields(client, auth_headers, services, field):
    files = {"audio": ("test.wav", silent_wav(100))} if field != "audio" else {}
    data = {key: value for key, value in {"source_lang": "en", "target_lang": "ko"}.items() if key != field}
    response = await client.post("/api/translate/speech", headers=auth_headers, files=files, data=data)
    assert response.status_code == 422 and isinstance(response.json()["detail"], list)


# 빈 파일·음성이 아닌 바이트는 "Audio could not be decoded", 30초를 1 ms 넘는 음성은 "Audio is longer than 30
# seconds"로 거절된다. 이 검사는 인식 모델을 부르기 전에 해야 하므로, 가짜 인식기가 불리면 503이 나도록 설정해
# 두고 422가 나오는지 본다.
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


# 정확히 30초는 허용한다(초과만 거절하는 경계).
async def test_exactly_30_seconds_accepted(client, auth_headers, services):
    response = await speech_request(client, auth_headers, silent_wav(30000))
    assert response.status_code == 201, response.text


# 모델 실패의 고정 응답: 인식 단계의 디코딩 실패·말 없음은 422(각각 다른 문구), 인식·번역 모델 오류는 503
# "Translation service is unavailable". 두 번 보내도 같은 답이고(가짜 실패가 한 번 쓰고 사라지지 않음),
# 로그에는 예외 종류만 있고 메시지("private ...")는 없으며, 기록과 음성 폴더도 생기지 않는지 확인한다.
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


# 원문과 번역문을 대신하는 표식. 로그·응답에 이 글자가 보이면 사용자 데이터가 샌 것이다.
SOURCE_MARK, TRANSLATION_MARK = "원문표식", "translation-mark"


# 실제 WhisperSpeechToText 안에 끼울 가짜 faster-whisper. CTranslate2처럼 호출 순간(lazy=False) 또는 조각을
# 읽는 도중(lazy=True)에 원문 표식을 담은 RuntimeError를 낸다.
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


# 실제 인식 어댑터의 예외 경계까지 거쳤을 때 엔진 오류가 500이 아니라 503이 되고, 로그에는 'ModelError <-
# RuntimeError' 종류 사슬만 남으며 원문 표식은 로그·응답 어디에도 없는지 확인한다(커밋 544f8d6).
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


# 오류 메시지에 입력 글자를 인용하는 가짜 토크나이저. 다른 예외(KeyError)를 처리하는 중에 또 인용
# 예외(ValueError)를 내, 예외 사슬의 두 메시지 모두 로그에서 빠지는지 보게 한다. B904 경고 끄기 표시는 암묵적
# 문맥 연결이 시험 대상이라 일부러 둔 것이다.
class QuotingTokenizer:
    """A tokenizer whose error quotes the text, raised while handling another error that quotes it too."""

    def encode(self, text, out_type):
        try:
            raise KeyError(f"no piece for {text}")
        except KeyError:
            raise ValueError(f"cannot encode {text}")  # noqa: B904 - the implicit context is the point

    def decode(self, pieces):
        return " ".join(pieces)


# 실제 Marian 번역 어댑터와 그 예외 경계를 쓰고 토크나이저·엔진만 바꿔, 번역 실패가 503이 되고 로그에는
# 'ModelError <- ValueError <- KeyError'(어느 단계에서 어떤 종류로 실패했는지)만 남는지, 원문 표식은 없는지
# 확인한다.
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


# 번역문에 표식을 넣는 가짜 번역기. 합성 실패 로그에 번역문이 새는지 보려는 것이다.
class MarkingTranslator:
    model_name = "marking-mt"

    def translate(self, text, source, target):
        return f"{TRANSLATION_MARK} of the text"


# 실제 합성 예외 경계(tts._guarded) 안에서 번역문을 인용하며 실패하는 가짜 합성기.
class QuotingSynthesis:
    """Speech synthesis whose error, inside the real model boundary, quotes the translation."""

    model_name = "quoting-tts"

    def synthesize(self, text, language):
        def front_end():
            raise ValueError(f"cannot read {text}")

        return tts._guarded(front_end)


# 합성이 실패해도 번역은 201로 돌려주고 tts_error를 싣는다(README 설계 판단). 로그에는 종류 사슬만 있고
# 번역문·원문 표식은 없어야 한다.
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


# 인식 결과가 비면 422 "No speech was recognized"이고 기록을 남기지 않는다.
async def test_fake_empty_recognition(client, auth_headers, services, db_session):
    models, _ = services
    models.stt.text = ""
    response = await speech_request(client, auth_headers)
    assert response.status_code == 422 and response.json() == {"detail": "No speech was recognized"}
    await assert_no_history(client, auth_headers, db_session)


# 합성 쪽 세 가지 실패(합성 모델 오류, 합성 모델 없음, 음성 파일 저장 실패)에서도 번역 기록은 201로 저장되고,
# 음성 id·합성 모델·합성 시간은 비며, 음성 행은 생기지 않는지 확인한다. 문구는 모델이 없으면 "Speech synthesis
# is not available", 나머지는 "Speech synthesis failed"다. 화면은 이때 브라우저 내장 음성으로 읽는다(README).
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


# 회귀 테스트(커밋 b79b29b): 인증의 사용자 조회 트랜잭션이 모델 스레드를 기다리기 전에 끝나, 번역 모델이 도는
# 동안 DB 연결 풀에서 빌린 연결이 0개인지 확인한다. 번역이 오래 걸려도 로그인·기록 요청이 연결을 기다리지 않게
# 한다(docs/api.md).
async def test_a_translation_holds_no_database_connection_during_model_work(
    client, auth_headers, services, db_engine
):
    """Login and history must not wait for translations (docs/api.md): the sign-in lookup's transaction ends
    before the request queues for the model thread, so no pooled connection sits idle meanwhile."""
    models, _ = services
    held = []

    class Watching(FakeTranslator):
        def translate(self, text, source, target):
            held.append(db_engine.sync_engine.pool.checkedout())
            return super().translate(text, source, target)

    app.dependency_overrides[get_models] = lambda: replace(models, translator=Watching())
    response = await text_request(client, auth_headers)
    assert response.status_code == 201, response.text
    assert held == [0]
    assert (await client.get("/api/history", headers=auth_headers)).json()["total"] == 1


# 모델 없이 띄운 서버(기본 get_models)에서 번역 요청은 503이고 기록이 없다.
async def test_missing_required_models(client, auth_headers, services, db_session):
    app.dependency_overrides.pop(get_models)
    response = await text_request(client, auth_headers)
    assert response.status_code == 503
    assert response.json() == {"detail": "Translation service is unavailable"}
    await assert_no_history(client, auth_headers, db_session)


# 번역·음성 경로는 모두 인증이 필요하고, 401에는 WWW-Authenticate: Bearer 헤더가 붙는다(docs/api.md 오류
# 응답).
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


# 음성 소유권: 다른 사용자가 내 음성 id로 요청하거나 없는 id를 요청하면 같은 404 "Audio not found"(존재 여부를
# 드러내지 않음), 남이 내 기록을 지우려 하면 404이고 내 음성은 그대로다. 내가 기록을 지우면 음성 파일·행·번역
# 행이 모두 사라지고 같은 음성 요청은 404, 형식이 틀린 id는 422다.
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


# DB 행은 있는데 디스크 파일이 없어진 경우도 같은 404로 답한다(500이 아님).
async def test_missing_disk_file_is_same_404(client, auth_headers, services):
    _, store = services
    result = (await text_request(client, auth_headers)).json()
    next(store.directory.iterdir()).unlink()
    response = await client.get(f"/api/audio/{result['audio_id']}", headers=auth_headers)
    assert response.status_code == 404 and response.json() == {"detail": "Audio not found"}


# 기록 삭제는 DB가 기준이다. 파일 삭제가 실패해도(사용 중인 파일) 204로 답하고 DB 행은 지워지며, 경고 로그만
# 남기고 남은 파일은 더 이상 내려주지 않는다(services/audio_store.py의 delete 설명).
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


# 업로드 한도는 multipart 머리 등 부가 바이트가 아니라 파일 바이트로 센다. 유효한 WAV 뒤를 0으로 채워 정확히
# 10MB면 통과(201), 1바이트 넘으면 413 "Audio file is larger than 10 MB"다.
@pytest.mark.parametrize("oversized", [False, True])
async def test_upload_limit_counts_file_bytes(client, auth_headers, services, oversized):
    # A valid WAV padded with trailing bytes verifies exact file bytes, excluding multipart overhead.
    wav = silent_wav(100)
    content = wav + b"\0" * (MAX_AUDIO_BYTES + int(oversized) - len(wav))
    response = await speech_request(client, auth_headers, content)
    assert response.status_code == (413 if oversized else 201), response.text
    if oversized:
        assert response.json() == {"detail": "Audio file is larger than 10 MB"}


# Content-Length 없이 조각으로 오는(chunked) 큰 업로드도, 서버가 10MB를 넘는 순간 읽기를 멈추고 413을 내는지
# 확인한다. 본문 생성기가 마지막 조각까지 소비되지 않았는지(consumed_tail)로 끝까지 받지 않았음을 본다. 큰
# 파일을 다 받은 뒤에 거절하는 것을 막는 처리다(routers/translate.py의 제한 파서).
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


# 실제 webm(opus, 48 kHz) 녹음을 PyAV로 만들어 보내, WAV가 아닌 압축 형식도 서버 디코딩을 통과하는지 확인한다.
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


# 경계(boundary)가 없거나 본문이 multipart가 아니면 400 "Invalid multipart request"로 답한다(500이 아님).
@pytest.mark.parametrize("content_type", ["multipart/form-data", "multipart/form-data; boundary=abc"])
async def test_malformed_multipart_returns_400(client, auth_headers, services, content_type):
    response = await client.post(
        "/api/translate/speech",
        content=b"not multipart",
        headers={**auth_headers, "Content-Type": content_type},
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid multipart request"}

import json
from pathlib import Path

import httpx
import pytest

from app.services import translation
from app.services.interfaces import ModelError
from app.services.translation import (
    DirectionalTranslator,
    OllamaTranslator,
    marian_source_tokens,
    nllb_source_tokens,
    opus_preprocess,
    split_sentences,
)
from tests.fakes import FakeTranslator


class SplitProcessor:
    """Stands in for SentencePiece: one piece per word."""

    def encode(self, text, out_type):
        assert out_type is str
        return text.split()

    def decode(self, pieces):
        return " ".join(pieces)


class FakeEngine:
    """Stands in for the CTranslate2 model: returns fixed pieces or raises."""

    def __init__(self, result=None, error=None):
        self.result, self.error = result, error

    def generate(self, tokens, target_prefix=None):
        if self.error:
            raise self.error
        return self.result


class EchoEngine:
    """Returns the source pieces in capitals as the translation and records each call."""

    def __init__(self):
        self.calls = []

    def generate(self, tokens, target_prefix=None):
        self.calls.append(tokens)
        return [piece.upper() for piece in tokens[:-1]]  # without the end-of-sentence token


@pytest.fixture
def engine(monkeypatch):
    """Build the real translator classes around a fake engine and tokenizer, without model files."""
    holder = {}
    monkeypatch.setattr(translation, "_Ct2Model", lambda *args: holder["engine"])
    monkeypatch.setattr(translation, "load_sentencepiece", lambda path: SplitProcessor())
    return lambda fake: holder.update(engine=fake)


def test_marian_returns_the_decoded_translation(engine):
    engine(FakeEngine(result=["Hello", "world"]))
    translator = translation.MarianTranslator(Path("model"), "ko", "en")
    assert translator.translate("안녕", "ko", "en") == "Hello world"


@pytest.mark.parametrize(
    "fake", [FakeEngine(error=RuntimeError("CUDA out of memory")), FakeEngine(result=[])]
)
def test_marian_engine_failures_become_model_errors(engine, fake):
    engine(fake)
    with pytest.raises(ModelError):
        translation.MarianTranslator(Path("model"), "ko", "en").translate("안녕", "ko", "en")


@pytest.mark.parametrize(
    "fake", [FakeEngine(error=RuntimeError("CUDA out of memory")), FakeEngine(result=["eng_Latn"])]
)
def test_nllb_engine_failures_become_model_errors(engine, fake):
    engine(fake)
    with pytest.raises(ModelError):
        translation.NllbTranslator(Path("model")).translate("안녕", "ko", "en")


def test_opus_preprocess_follows_the_training_script():
    assert opus_preprocess("“안녕”，  세상！") == '"안녕", 세상!'
    assert opus_preprocess("끝。다음") == "끝. 다음"
    assert opus_preprocess(" a​b\tc ") == "a b c"  # zero-width and control characters become spaces


@pytest.mark.parametrize(
    ("text", "sentences"),
    [
        ("비가 왔다. 그래서 집에 있었다.", ["비가 왔다.", "그래서 집에 있었다."]),
        ("Is it? Yes!\nGood.", ["Is it?", "Yes!", "Good."]),
        ("첫째。 둘째？ 셋째！", ["첫째。", "둘째？", "셋째！"]),
        (
            "The U.N. met Mr. Smith and J. Doe, e.g. at 3.5 km. Then left.",
            ["The U.N. met Mr. Smith and J. Doe, e.g. at 3.5 km.", "Then left."],
        ),
        ("  한 문장  ", ["한 문장"]),
        ("   ", []),
    ],
)
def test_split_sentences(text, sentences):
    assert split_sentences(text) == sentences


def test_marian_by_sentence_translates_each_sentence_on_its_own(engine):
    echo = EchoEngine()
    engine(echo)
    translator = translation.MarianTranslator(Path("model"), "en", "ko", by_sentence=True)
    result = translator.translate("It rained. Dr. Kim left early!  Why?", "en", "ko")
    assert result == "IT RAINED. DR. KIM LEFT EARLY! WHY?"
    assert len(echo.calls) == 3


def test_marian_translates_the_whole_input_by_default(engine):
    echo = EchoEngine()
    engine(echo)
    translation.MarianTranslator(Path("model"), "en", "ko").translate("One. Two.", "en", "ko")
    assert len(echo.calls) == 1


def test_source_tokens_add_language_code_and_end_of_sentence():
    assert marian_source_tokens(SplitProcessor(), "Hello，world") == ["Hello,world", "</s>"]
    assert nllb_source_tokens(SplitProcessor(), "안녕 세상", "ko") == ["kor_Hang", "안녕", "세상", "</s>"]


def test_directional_translator_routes_each_direction():
    translator = DirectionalTranslator({("ko", "en"): FakeTranslator()})
    assert translator.translate("안녕", "ko", "en") == "[ko->en] 안녕"
    assert "ko->en" in translator.model_name
    with pytest.raises(ModelError, match="no translation model"):
        translator.translate("hello", "en", "ko")
    with pytest.raises(ModelError):
        translator.translate("안녕", "ko", "ko")
    with pytest.raises(ModelError):
        translator.translate("   ", "ko", "en")


def ollama_with(handler) -> tuple[OllamaTranslator, list[dict]]:
    sent: list[dict] = []

    def record(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return handler(request)

    translator = OllamaTranslator("qwen2.5:7b-instruct")
    translator._client = httpx.Client(base_url="http://ollama.test", transport=httpx.MockTransport(record))
    return translator, sent


def test_ollama_translator_uses_a_fixed_prompt_and_returns_the_reply():
    translator, sent = ollama_with(lambda _: httpx.Response(200, json={"message": {"content": " Hello \n"}}))
    assert translator.translate("안녕하세요", "ko", "en") == "Hello"
    payload = sent[0]
    assert payload["options"]["temperature"] == 0
    assert payload["stream"] is False
    assert "Korean text into English" in payload["messages"][0]["content"]
    assert payload["messages"][1] == {"role": "user", "content": "안녕하세요"}


@pytest.mark.parametrize(
    "reply",
    [
        httpx.Response(500, json={"error": "model crashed"}),
        httpx.Response(200, json={"message": {"content": "   "}}),
        httpx.Response(200, json={"message": {"content": None}}),
        httpx.Response(200, json={"message": {"content": 123}}),
        httpx.Response(200, json={"message": None}),
        httpx.Response(200, text="not json"),
    ],
)
def test_ollama_failures_become_model_errors(reply):
    translator, _ = ollama_with(lambda _: reply)
    with pytest.raises(ModelError):
        translator.translate("hello", "en", "ko")


def test_ollama_connection_error_becomes_model_error():
    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    translator, _ = ollama_with(refuse)
    with pytest.raises(ModelError):
        translator.translate("hello", "en", "ko")

import json

import httpx
import pytest

from app.services.interfaces import ModelError
from app.services.translation import (
    DirectionalTranslator,
    OllamaTranslator,
    marian_source_tokens,
    nllb_source_tokens,
    opus_preprocess,
)
from tests.fakes import FakeTranslator


class SplitProcessor:
    """Stands in for SentencePiece: one piece per word."""

    def encode(self, text, out_type):
        assert out_type is str
        return text.split()


def test_opus_preprocess_follows_the_training_script():
    assert opus_preprocess("“안녕”，  세상！") == '"안녕", 세상!'
    assert opus_preprocess("끝。다음") == "끝. 다음"
    assert opus_preprocess(" a​b\tc ") == "a b c"  # zero-width and control characters become spaces


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
        httpx.Response(200, text="not json"),
    ],
)
def test_ollama_failures_become_model_errors(reply):
    translator, _ = ollama_with(lambda _: reply)
    with pytest.raises(ModelError):
        translator.translate("hello", "en", "ko")

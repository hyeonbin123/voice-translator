import json
from pathlib import Path

import httpx
import pytest

from app.services import translation
from app.services.interfaces import ModelError
from app.services.translation import (
    DirectionalTranslator,
    HyMtTranslator,
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
        ("Meet\nDr. Kim. Next.", ["Meet\nDr. Kim.", "Next."]),
        ("Ask\tMr. Lee first. Then go.", ["Ask\tMr. Lee first.", "Then go."]),
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


@pytest.mark.parametrize(
    ("text", "calls"),
    [
        ("Meet\nDr. Kim. Next.", [["Meet", "Dr.", "Kim.", "</s>"], ["Next.", "</s>"]]),
        ("Ask\tMr. Lee first. Then go.", [["Ask", "Mr.", "Lee", "first.", "</s>"], ["Then", "go.", "</s>"]]),
        ("Met\nJ. Doe. Bye.", [["Met", "J.", "Doe.", "</s>"], ["Bye.", "</s>"]]),
    ],
)
def test_marian_by_sentence_keeps_abbreviations_after_any_white_space(engine, text, calls):
    # T27/T31: what reaches the engine, sentence by sentence, through the real adapter.
    echo = EchoEngine()
    engine(echo)
    translation.MarianTranslator(Path("model"), "en", "ko", by_sentence=True).translate(text, "en", "ko")
    assert echo.calls == calls


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


# Hy-MT2 served by Ollama (T83, docs/experiments.md 15)

HY_MT_ENGLISH_PROMPT = (
    "Translate the following text into Korean. Note that you should only output the translated result "
    "without any additional explanation:\n\n"
)


def hymt_with(handler, **kwargs) -> tuple[HyMtTranslator, list[tuple[str, dict | None]]]:
    sent: list[tuple[str, dict | None]] = []

    def record(request: httpx.Request) -> httpx.Response:
        sent.append((request.url.path, json.loads(request.content) if request.content else None))
        return handler(request)

    translator = HyMtTranslator("hy-mt2:1.8b-q8_0", base_url="http://ollama.test", **kwargs)
    translator._client = httpx.Client(base_url="http://ollama.test", transport=httpx.MockTransport(record))
    return translator, sent


def chat_reply(content, done_reason="stop", **extra) -> httpx.Response:
    body = {
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": done_reason,
        "prompt_eval_count": 40,
        "eval_count": 7,
        **extra,
    }
    return httpx.Response(200, json=body)


def test_hymt_sends_the_model_card_prompt_as_the_only_message():
    translator, sent = hymt_with(lambda _: chat_reply(" 안녕하세요. \n"))
    assert translator.translate("Hello.", "en", "ko") == "안녕하세요."
    path, payload = sent[0]
    assert path == "/api/chat"
    # The model card's "Default Translation" English prompt, no system prompt (the model has no default one).
    assert payload["messages"] == [{"role": "user", "content": HY_MT_ENGLISH_PROMPT + "Hello."}]
    assert payload["model"] == "hy-mt2:1.8b-q8_0"
    assert payload["stream"] is False
    assert payload["keep_alive"] == -1
    # Greedy, the card's repetition penalty, the opus output cap of 256 tokens, a fixed context size.
    assert payload["options"] == {
        "temperature": 0,
        "repeat_penalty": 1.05,
        "num_predict": 256,
        "num_ctx": 2048,
    }
    assert translator.model_name == "ollama/hy-mt2:1.8b-q8_0"


def test_hymt_names_the_target_language_in_english():
    translator, sent = hymt_with(lambda _: chat_reply("Hello."))
    translator.translate("안녕하세요.", "ko", "en")
    assert sent[0][1]["messages"][0]["content"].startswith("Translate the following text into English. ")
    assert sent[0][1]["messages"][0]["content"].endswith(":\n\n안녕하세요.")


def test_hymt_by_sentence_translates_each_sentence_on_its_own():
    replies = iter(["안녕.", "잘 지내?"])
    translator, sent = hymt_with(lambda _: chat_reply(next(replies)), by_sentence=True)
    assert translator.translate("Hi there. How are you?", "en", "ko") == "안녕. 잘 지내?"
    assert [payload["messages"][0]["content"] for _, payload in sent] == [
        HY_MT_ENGLISH_PROMPT + "Hi there.",
        HY_MT_ENGLISH_PROMPT + "How are you?",
    ]
    assert translator.model_name == "ollama/hy-mt2:1.8b-q8_0/split"


def test_hymt_keeps_the_details_of_the_last_translation_for_evaluation():
    translator, _ = hymt_with(lambda _: chat_reply("안녕.", eval_count=3), by_sentence=True)
    translator.translate("Hi. Bye.", "en", "ko")
    translator.translate("Hi.", "en", "ko")
    assert translator.calls == [{"done_reason": "stop", "prompt_eval_count": 40, "eval_count": 3}]


def test_hymt_keeps_a_reply_cut_at_the_output_cap():
    # Like opus at max_decoding_length 256: a cut translation is still the translation.
    translator, _ = hymt_with(lambda _: chat_reply("아주 긴 번역", done_reason="length"))
    assert translator.translate("A very long text.", "en", "ko") == "아주 긴 번역"
    assert translator.calls[0]["done_reason"] == "length"


@pytest.mark.parametrize(
    "reply",
    [
        httpx.Response(500, json={"error": "model crashed"}),
        httpx.Response(404, json={"error": "model not found"}),
        chat_reply("   "),
        chat_reply(None),
        chat_reply(123),
        httpx.Response(200, json={"message": None}),
        httpx.Response(200, text="not json"),
    ],
)
def test_hymt_failures_become_model_errors(reply):
    translator, _ = hymt_with(lambda _: reply)
    with pytest.raises(ModelError):
        translator.translate("hello", "en", "ko")


def test_hymt_connection_error_becomes_model_error():
    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    translator, _ = hymt_with(refuse)
    with pytest.raises(ModelError):
        translator.translate("hello", "en", "ko")


def ollama_server(tags_digest="sha256-of-the-build", show_status=200):
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/show":
            return httpx.Response(show_status, json={"template": "..."})
        if request.url.path == "/api/tags":
            models = [
                {"name": "other:1b", "digest": "other"},
                {"name": "hy-mt2:1.8b-q8_0", "digest": tags_digest},
            ]
            return httpx.Response(200, json={"models": models})
        if request.url.path == "/api/generate":
            return httpx.Response(200, json={"done": True})
        return httpx.Response(404)

    return handle


def test_hymt_prepare_loads_the_model_to_stay_and_reports_its_build():
    translator, sent = hymt_with(ollama_server())
    assert translator.prepare() == "sha256-of-the-build"
    assert translator.digest == "sha256-of-the-build"
    generate = [payload for path, payload in sent if path == "/api/generate"]
    assert generate == [{"model": "hy-mt2:1.8b-q8_0", "keep_alive": -1}]


def test_hymt_prepare_fails_when_ollama_lacks_the_model():
    translator, sent = hymt_with(ollama_server(show_status=404))
    with pytest.raises(ModelError, match="hymt_setup"):
        translator.prepare()
    assert not any(path == "/api/generate" for path, _ in sent)


def test_hymt_prepare_fails_while_ollama_is_down():
    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    translator, _ = hymt_with(refuse)
    with pytest.raises(ModelError):
        translator.prepare()


def test_hymt_prepare_refuses_another_build_of_the_model():
    translator, sent = hymt_with(
        ollama_server(tags_digest="another-build"), expected_digest="the-measured-build"
    )
    with pytest.raises(RuntimeError, match="another-build"):
        translator.prepare()
    assert not any(path == "/api/generate" for path, _ in sent)

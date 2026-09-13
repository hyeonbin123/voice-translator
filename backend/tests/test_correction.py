import ast
import json
from pathlib import Path

import httpx
import pytest

from app.services import correction
from app.services.correction import OllamaCorrector

EVAL_SCRIPT = Path(__file__).resolve().parents[1] / "eval" / "typo_eval.py"


def corrector_with(handler) -> tuple[OllamaCorrector, list[tuple[str, dict]]]:
    sent: list[tuple[str, dict]] = []

    def record(request: httpx.Request) -> httpx.Response:
        sent.append((request.url.path, json.loads(request.content)))
        return handler(request)

    corrector = OllamaCorrector("qwen2.5:1.5b-instruct")
    corrector._client = httpx.Client(base_url="http://ollama.test", transport=httpx.MockTransport(record))
    return corrector, sent


def test_the_server_uses_the_instruction_that_was_measured():
    # Read without importing: eval/typo_eval.py needs the eval dependency group.
    tree = ast.parse(EVAL_SCRIPT.read_text(encoding="utf-8"))
    measured = next(
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "SYSTEM"
    )
    assert correction.SYSTEM == measured


def test_only_english_is_corrected_by_default():
    corrector = OllamaCorrector("qwen2.5:1.5b-instruct")
    assert corrector.corrects("en") and not corrector.corrects("ko")
    assert corrector.model_name == "ollama/qwen2.5:1.5b-instruct"


def test_correction_sends_the_fixed_instruction_and_returns_the_reply():
    corrector, sent = corrector_with(
        lambda _: httpx.Response(
            200, json={"message": {"content": " I have a cat.\n"}, "done_reason": "stop"}
        )
    )
    assert corrector.correct("I hvae a cat.", "en") == "I have a cat."
    path, payload = sent[0]
    assert path == "/api/chat"
    assert payload["options"]["temperature"] == 0 and payload["stream"] is False
    assert payload["keep_alive"] == -1
    assert "in the user's English text" in payload["messages"][0]["content"]
    assert payload["messages"][1] == {"role": "user", "content": "I hvae a cat."}


@pytest.mark.parametrize(
    "reply",
    [
        httpx.Response(500, json={"error": "model crashed"}),
        httpx.Response(404, json={"error": "model not found"}),
        httpx.Response(200, json={"message": {"content": "   "}}),
        httpx.Response(200, json={"message": {"content": None}}),
        httpx.Response(200, json={"message": None}),
        httpx.Response(200, json={}),
        httpx.Response(200, text="not json"),
        # Cut off at num_predict: translating it would drop the end of the text.
        httpx.Response(200, json={"message": {"content": "I have a"}, "done_reason": "length"}),
    ],
)
def test_a_failed_correction_gives_none_and_never_logs_the_text(reply, caplog):
    corrector, _ = corrector_with(lambda _: reply)
    assert corrector.correct("private words typed here", "en") is None
    assert "Typo correction" in caplog.text
    assert "private words typed here" not in caplog.text


@pytest.mark.parametrize("error", [httpx.ConnectError, httpx.ReadTimeout])
def test_unreachable_or_slow_ollama_gives_none(error):
    def fail(request):
        raise error("no answer", request=request)

    corrector, _ = corrector_with(fail)
    assert corrector.correct("I hvae a cat.", "en") is None


def test_prepare_pulls_a_missing_model_loads_it_to_stay_and_corrects_once():
    replies = {
        "/api/show": httpx.Response(404, json={"error": "model not found"}),
        "/api/pull": httpx.Response(200, json={"status": "success"}),
        "/api/generate": httpx.Response(200, json={"done": True}),
        "/api/chat": httpx.Response(200, json={"message": {"content": "Hello."}, "done_reason": "stop"}),
    }
    corrector, sent = corrector_with(lambda request: replies[request.url.path])
    corrector.prepare()
    assert [path for path, _ in sent] == ["/api/show", "/api/pull", "/api/generate", "/api/chat"]
    assert sent[1][1] == {"model": "qwen2.5:1.5b-instruct", "stream": False}
    assert sent[2][1] == {"model": "qwen2.5:1.5b-instruct", "keep_alive": -1}
    # The first correction runs at startup with the same request the server sends later.
    assert sent[3][1]["messages"][0]["content"] == correction.SYSTEM.format(language="English")


def test_prepare_does_not_pull_a_model_that_is_there():
    corrector, sent = corrector_with(lambda _: httpx.Response(200, json={}))
    corrector.prepare()
    assert [path for path, _ in sent] == ["/api/show", "/api/generate", "/api/chat"]


def test_prepare_raises_when_ollama_fails():
    corrector, _ = corrector_with(lambda _: httpx.Response(500, json={"error": "broken"}))
    with pytest.raises(httpx.HTTPStatusError):
        corrector.prepare()

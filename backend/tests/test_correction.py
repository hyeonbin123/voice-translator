import ast
import json
import logging
import threading
from pathlib import Path

import httpx
import pytest

from app.services import correction
from app.services.correction import OllamaCorrector

EVAL_SCRIPT = Path(__file__).resolve().parents[1] / "eval" / "typo_eval.py"
FINISHED = {"done": True, "done_reason": "stop"}


def corrector_with(handler, **options) -> tuple[OllamaCorrector, list[tuple[str, dict]]]:
    sent: list[tuple[str, dict]] = []

    def record(request: httpx.Request) -> httpx.Response:
        sent.append((request.url.path, json.loads(request.content)))
        return handler(request)

    corrector = OllamaCorrector("qwen2.5:1.5b-instruct", **options)
    corrector._client.close()  # replaced by a client that only reaches the handler
    corrector._client = httpx.Client(base_url="http://ollama.test", transport=httpx.MockTransport(record))
    return corrector, sent


class SteppedLock:
    """Pauses the preparation thread at its n-th use of the corrector's lock, to hit one interleaving."""

    def __init__(self, pause_at: int) -> None:
        self._lock = threading.Lock()
        self.pause_at = pause_at
        self.uses = 0
        self.waiting = threading.Event()
        self.go = threading.Event()

    def __enter__(self) -> None:
        if threading.current_thread().name == "typo-correction-prepare":
            self.uses += 1
            if self.uses == self.pause_at:
                self.waiting.set()
                assert self.go.wait(5)
        self._lock.acquire()

    def __exit__(self, *exc) -> None:
        self._lock.release()


@pytest.mark.parametrize(
    ("pause_at", "was_ready"), [(1, False), (2, True)], ids=["before-turning-on", "at-worker-end"]
)
def test_close_racing_the_worker_closes_the_client_and_never_turns_on_after(pause_at, was_ready, caplog):
    # T41: close() lands while the worker is paused right before its readiness check, or right before its
    # final clean-up (where the client used to stay open).
    caplog.set_level(logging.INFO, logger="app.services.correction")
    corrector, _ = corrector_with(
        lambda _: httpx.Response(200, json={"message": {"content": "Hello."}, **FINISHED})
    )
    lock = corrector._lock = SteppedLock(pause_at)
    thread = corrector.start()
    assert lock.waiting.wait(5)
    corrector.close()
    lock.go.set()
    thread.join(5)
    assert not thread.is_alive()
    assert corrector._client.is_closed and not corrector.corrects("en")
    assert ("is ready" in caplog.text) is was_ready


def test_close_without_a_worker_closes_the_client():
    corrector, sent = corrector_with(
        lambda _: httpx.Response(200, json={"message": {"content": "Hi."}, **FINISHED})
    )
    corrector.close()
    assert corrector._client.is_closed
    assert corrector.correct("I hvae a cat.", "en") is None and sent == []


def test_the_server_uses_the_instruction_that_was_measured():
    # Read without importing: eval/typo_eval.py needs the eval dependency group.
    tree = ast.parse(EVAL_SCRIPT.read_text(encoding="utf-8"))
    measured = next(
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "SYSTEM"
    )
    assert correction.SYSTEM == measured


def test_start_does_not_wait_and_corrects_english_only_once_ready():
    gate = threading.Event()

    def slow_ollama(_):
        assert gate.wait(5)
        return httpx.Response(
            200, json={"message": {"content": "Hello."}, "done": True, "done_reason": "stop"}
        )

    corrector, _ = corrector_with(slow_ollama)
    assert corrector.model_name == "ollama/qwen2.5:1.5b-instruct"
    thread = corrector.start()
    # Startup goes on while Ollama is still busy, and requests do not wait for it.
    assert not corrector.corrects("en")
    gate.set()
    thread.join(5)
    assert corrector.corrects("en") and not corrector.corrects("ko")


def test_a_failed_start_is_logged_once_and_retried_until_ollama_answers(caplog):
    caplog.set_level(logging.INFO, logger="app.services.correction")
    refusals = iter(range(2))

    def late_ollama(request):
        if next(refusals, None) is not None:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(
            200, json={"message": {"content": "Hello."}, "done": True, "done_reason": "stop"}
        )

    corrector, _ = corrector_with(late_ollama, retry_s=0.01)
    corrector.start().join(5)
    assert corrector.corrects("en")
    assert caplog.text.count("is not ready") == 1
    assert "is ready" in caplog.text


def test_close_stops_retrying():
    def down(request):
        raise httpx.ConnectError("connection refused", request=request)

    corrector, _ = corrector_with(down, retry_s=0.01)
    thread = corrector.start()
    corrector.close()
    thread.join(5)
    assert not thread.is_alive() and not corrector.corrects("en")


def test_every_preparation_call_has_a_time_limit():
    timeouts: list[dict] = []

    def record(request: httpx.Request) -> httpx.Response:
        timeouts.append(request.extensions["timeout"])
        if request.url.path == "/api/show":
            return httpx.Response(404, json={"error": "model not found"})
        return httpx.Response(
            200, json={"message": {"content": "Hello."}, "done": True, "done_reason": "stop"}
        )

    corrector = OllamaCorrector("qwen2.5:1.5b-instruct", timeout_s=10, prepare_timeout_s=60)
    corrector._client.close()
    corrector._client = httpx.Client(
        base_url="http://ollama.test", timeout=10, transport=httpx.MockTransport(record)
    )
    corrector.prepare()
    assert len(timeouts) == 4  # show, pull, load, first correction
    assert all(value is not None for timeout in timeouts for value in timeout.values())
    # The download may take minutes, but a stalled Ollama cannot hold the thread forever.
    assert [timeout["read"] for timeout in timeouts] == [10, 60, 60, 60]


def test_correction_sends_the_fixed_instruction_and_returns_the_reply():
    corrector, sent = corrector_with(
        lambda _: httpx.Response(200, json={"message": {"content": " I have a cat.\n"}, **FINISHED})
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


@pytest.mark.parametrize(
    "body",
    [
        {"message": {"content": "I have a cat."}, "done": False},
        {"message": {"content": "I have a cat."}, "done": True, "done_reason": "load"},
        {"message": {"content": "I have a cat."}},
        {"message": {"content": "I cannot help with that request."}, **FINISHED},
        {"message": {"content": "고양이가 있습니다."}, **FINISHED},
        {
            "message": {"content": "Sure! Here is the corrected text: I have a cat. Anything else?"},
            **FINISHED,
        },
    ],
)
def test_unfinished_or_unlike_replies_leave_the_text_as_typed(body, caplog):
    corrector, _ = corrector_with(lambda _: httpx.Response(200, json=body))
    assert corrector.correct("I hvae a cat.", "en") is None
    assert "Typo correction" in caplog.text and "I hvae a cat." not in caplog.text


@pytest.mark.parametrize(
    ("typed", "fixed"),
    [
        ("I hvae a cat.", "I have a cat."),
        (
            "Argentinais well known for haviing oneof the bestpolo teams an players inthe world.",
            "Argentina is well known for having one of the best polo teams and players in the world.",
        ),
    ],
)
def test_real_corrections_pass_the_safeguard(typed, fixed):
    corrector, _ = corrector_with(
        lambda _: httpx.Response(200, json={"message": {"content": fixed}, **FINISHED})
    )
    assert corrector.correct(typed, "en") == fixed


def test_long_corrections_are_not_rated_unlike():
    # validation id 1524 (docs/experiments.md 6): with difflib's autojunk this good correction scored 0.152.
    typed = (
        "There arw a lot og social and political effects such as the use of metric system,a shift from "
        "absolutism to republicanism, nationalism and thebeliefthecountry belongsto the people not to one "
        "sole ruler."
    )
    fixed = (
        "There are a lot of social and political effects, such as the use of the metric system, a shift "
        "from absolutism to republicanism, nationalism, and the belief that the country belongs to the "
        "people, not to one sole ruler."
    )
    assert correction.looks_like_a_correction(typed, fixed, "en")


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

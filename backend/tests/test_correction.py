# [파일 역할] 글자로 입력한 영어의 오타 교정기(app/services/correction.py의 OllamaCorrector)를 검증하는 테스트
# 묶음이다. 실제 Ollama 없이 httpx.MockTransport로 응답을 꾸며 넣는다.
# [무엇을 지키나]
# - 측정한 그대로 쓰는지: 서버 지시문이 T32에서 측정한 eval/typo_eval.py의 지시문과 같은지, temperature
#   0·keep_alive -1 요청인지(docs/experiments.md 6절).
# - 실패해도 번역은 되는지: Ollama 오류·꺼짐·느림, 잘린 답, 빈 답, 끝나지 않은 답, 거절문·번역해 버린
#   답·설명을 붙인 답이면 None을 돌려줘 입력 그대로 번역하게 한다(T38 안전장치).
# - 개인 정보: 실패를 로그에 남겨도 사용자가 입력한 글은 남기지 않는다.
# - 시작과 종료: 서버 시작을 막지 않고 뒤에서 준비하며(모델이 없으면 내려받음), 준비 전에는 교정하지 않고,
#   종료(close)가 준비 스레드와 겹쳐도 클라이언트가 닫히고 다시 켜지지 않는다.

import ast
import json
import logging
import threading
from pathlib import Path

# httpx: MockTransport로 Ollama HTTP 응답을 흉내 낸다. 테스트 대상은 서버 모듈 correction과 그 안의
# OllamaCorrector다.
import httpx
import pytest

from app.services import correction
from app.services.correction import OllamaCorrector

# EVAL_SCRIPT: 측정에 쓴 지시문이 들어 있는 eval/typo_eval.py 경로. FINISHED: Ollama가 정상으로 끝냈다는 응답
# 필드(done true, done_reason 'stop'). 안전장치는 이 둘이 모두 있어야 교정문을 쓴다.
EVAL_SCRIPT = Path(__file__).resolve().parents[1] / "eval" / "typo_eval.py"
FINISHED = {"done": True, "done_reason": "stop"}


# 실제 HTTP 대신 handler 함수가 답하는 교정기를 만든다. 보낸 요청(경로, JSON 본문)을 sent에 쌓아 테스트가
# 무엇을 보냈는지 확인할 수 있다. 생성자가 만든 원래 클라이언트는 닫고 바꿔 끼운다.
def corrector_with(handler, **options) -> tuple[OllamaCorrector, list[tuple[str, dict]]]:
    sent: list[tuple[str, dict]] = []

    def record(request: httpx.Request) -> httpx.Response:
        sent.append((request.url.path, json.loads(request.content)))
        return handler(request)

    corrector = OllamaCorrector("qwen2.5:1.5b-instruct", **options)
    corrector._client.close()  # handler에만 닿는 클라이언트로 바꾼다
    corrector._client = httpx.Client(base_url="http://ollama.test", transport=httpx.MockTransport(record))
    return corrector, sent


# 교정기의 Lock 대신 넣는 Lock. 준비 스레드(이름 typo-correction-prepare)가 n번째로 Lock을 쓰려는 순간 멈춰,
# 테스트가 그 사이에 close()를 끼워 넣을 수 있게 한다. 스레드 경합은 우연히는 재현이 어려우므로 특정 끼어들기
# 순서를 확정적으로 만든다.
# 준비 스레드의 Lock 사용: 1번째 = 준비 성공 뒤 '아직 닫히지 않았으면 준비 완료로 표시', 2번째 = 끝날 때의
# 정리(finally).
class SteppedLock:
    """준비 스레드가 교정기 Lock을 n번째로 쓸 때 멈춰, 정해진 끼어들기 순서 하나를 만든다."""

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


# 회귀 테스트: 준비가 끝나는 순간 서버 종료(close)가 겹치는 두 경우.
# - before-turning-on: 준비 완료 표시 직전에 close → 교정이 켜지지 않아야 한다('is ready' 로그 없음).
# - at-worker-end: 준비 완료 뒤 정리 직전에 close → 예전에는 이 순서에서 HTTP 클라이언트가 열린 채 남았다.
#   이제 준비 스레드의 정리가 닫아야 한다.
# 두 경우 모두 스레드가 끝나고, 클라이언트가 닫히고, 교정이 꺼져 있어야 한다.
@pytest.mark.parametrize(
    ("pause_at", "was_ready"), [(1, False), (2, True)], ids=["before-turning-on", "at-worker-end"]
)
def test_close_racing_the_worker_closes_the_client_and_never_turns_on_after(pause_at, was_ready, caplog):
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


# 준비 스레드를 시작하지 않은 채 close하면 close가 직접 클라이언트를 닫고, 그 뒤 교정 요청은 Ollama를 부르지
# 않고 None이어야 한다.
def test_close_without_a_worker_closes_the_client():
    corrector, sent = corrector_with(
        lambda _: httpx.Response(200, json={"message": {"content": "Hi."}, **FINISHED})
    )
    corrector.close()
    assert corrector._client.is_closed
    assert corrector.correct("I hvae a cat.", "en") is None and sent == []


# 서버 지시문(correction.SYSTEM)이 T32에서 측정한 지시문(eval/typo_eval.py의 SYSTEM)과 글자 하나까지 같아야
# 한다. 지시문을 바꾸면 측정 결과(오타 평균 +7.6점 등)가 더는 서버를 설명하지 못하기 때문이다. typo_eval.py는
# eval 그룹 패키지가 필요해 import하지 않고 AST로 상수만 읽는다.
def test_the_server_uses_the_instruction_that_was_measured():
    tree = ast.parse(EVAL_SCRIPT.read_text(encoding="utf-8"))
    measured = next(
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "SYSTEM"
    )
    assert correction.SYSTEM == measured


# start는 준비를 기다리지 않고 바로 돌아와야 하고(Ollama가 느려도 서버 시작이 막히지 않음), 준비가 끝나기
# 전에는 교정하지 않으며, 끝난 뒤에는 영어만 교정해야 한다(한국어 입력은 교정하지 않는다는 6절 결정).
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
    # Ollama가 아직 바쁜 동안에도 서버 시작은 진행되고, 요청은 준비를 기다리지 않는다.
    assert not corrector.corrects("en")
    gate.set()
    thread.join(5)
    assert corrector.corrects("en") and not corrector.corrects("ko")


# Ollama가 처음 두 번 연결을 거부하면 준비 스레드가 재시도해 결국 준비되고, '준비 안 됨' 경고는 재시도마다가
# 아니라 한 번만 남아야 한다(로그가 쌓이지 않게). retry_s를 0.01초로 줄여 빨리 돈다.
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


# Ollama가 계속 꺼져 있어도 close하면 재시도 대기를 깨고 스레드가 끝나야 한다(종료가 멈추지 않음).
def test_close_stops_retrying():
    def down(request):
        raise httpx.ConnectError("connection refused", request=request)

    corrector, _ = corrector_with(down, retry_s=0.01)
    thread = corrector.start()
    corrector.close()
    thread.join(5)
    assert not thread.is_alive() and not corrector.corrects("en")


# 준비 단계의 요청 4개(모델 확인, 내려받기, 올리기, 첫 교정)에 모두 시간 제한이 있어야 한다. 확인은 일반
# 제한(10초), 오래 걸릴 수 있는 셋은 prepare_timeout_s(60초)다. 멈춘 Ollama가 준비 스레드를 영원히 잡지 못하게
# 한다.
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
    assert len(timeouts) == 4  # 모델 확인, 내려받기, 올리기, 첫 교정
    assert all(value is not None for timeout in timeouts for value in timeout.values())
    # 내려받기는 몇 분 걸릴 수 있지만, 멈춘 Ollama가 스레드를 영원히 잡아 둘 수는 없다.
    assert [timeout["read"] for timeout in timeouts] == [10, 60, 60, 60]


# 교정 요청 형식: /api/chat, temperature 0, 스트리밍 없음, keep_alive -1(모델을 계속 올려 둠), 시스템 지시문에
# 언어 이름(English), 사용자 메시지는 입력 그대로. 답은 앞뒤 공백을 지워 돌려준다.
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


# 교정이 실패하는 여러 응답(서버 오류, 모델 없음, 빈 답, 형식이 깨진 JSON, 256토큰에서 잘린 답)은 모두
# None이어야 하고, 실패 로그는 남기되 사용자가 입력한 글은 로그에 없어야 한다. 잘린 답을 쓰면 문장 끝이 빠진
# 채 번역되기 때문이다.
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
        # num_predict에서 잘린 답: 이것을 번역하면 글 끝이 빠진다.
        httpx.Response(200, json={"message": {"content": "I have a"}, "done_reason": "length"}),
    ],
)
def test_a_failed_correction_gives_none_and_never_logs_the_text(reply, caplog):
    corrector, _ = corrector_with(lambda _: reply)
    assert corrector.correct("private words typed here", "en") is None
    assert "Typo correction" in caplog.text
    assert "private words typed here" not in caplog.text


# T38 안전장치: 끝나지 않은 답(done false, done_reason이 stop이 아님, 필드 없음), 거절문, 한국어로 번역해 버린
# 답, 설명을 덧붙인 답은 모두 버리고 입력 그대로 번역해야 한다. 거절문·설명형은 입력과의 글자 유사도(0.5
# 미만)로, 번역해 버린 답은 한글 포함으로 걸린다(6절 '교정 답 안전장치').
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


# 반대로 제대로 고친 교정은 안전장치를 통과해야 한다. 두 번째는 오타가 많은 긴 문장으로, 유사도 문턱이 정상
# 교정을 버리지 않는지 본다.
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


# validation id 1524의 실제 사례: 200자가 넘는 문장은 difflib의 autojunk가 자주 나오는 글자를 빼 유사도가
# 0.152로 낮게 나와 좋은 교정이 버려졌다. autojunk=False로 고친 뒤 통과하는지 확인하는 회귀 테스트다(6절
# '안전장치 확인: validation 1차').
def test_long_corrections_are_not_rated_unlike():
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


# Ollama에 연결할 수 없거나 응답이 시간 제한을 넘으면 예외를 밖으로 내지 않고 None(입력 그대로 번역)이어야
# 한다.
@pytest.mark.parametrize("error", [httpx.ConnectError, httpx.ReadTimeout])
def test_unreachable_or_slow_ollama_gives_none(error):
    def fail(request):
        raise error("no answer", request=request)

    corrector, _ = corrector_with(fail)
    assert corrector.correct("I hvae a cat.", "en") is None


# 모델이 없을 때(첫 compose 시작) 준비 순서: /api/show가 404 → /api/pull로 내려받기 → /api/generate에
# keep_alive -1로 올려 두기 → /api/chat으로 교정 한 번. 첫 교정이 서버가 나중에 보낼 것과 같은 지시문인지도
# 본다. 시작 때 교정을 한 번 하는 것은 첫 compose 시작에서 처음 교정 세 번이 10초 제한을 넘은 일이 있어 그
# 비용을 미리 치르려는 것이다(correction.py의 주석).
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
    assert sent[3][1]["messages"][0]["content"] == correction.SYSTEM.format(language="English")


# 모델이 이미 있으면 내려받기를 건너뛴다.
def test_prepare_does_not_pull_a_model_that_is_there():
    corrector, sent = corrector_with(lambda _: httpx.Response(200, json={}))
    corrector.prepare()
    assert [path for path, _ in sent] == ["/api/show", "/api/generate", "/api/chat"]


# prepare 자체는 Ollama 오류를 예외로 올려야 한다. 그래야 준비 스레드가 실패를 알고 나중에 다시 시도한다.
def test_prepare_raises_when_ollama_fails():
    corrector, _ = corrector_with(lambda _: httpx.Response(500, json={"error": "broken"}))
    with pytest.raises(httpx.HTTPStatusError):
        corrector.prepare()

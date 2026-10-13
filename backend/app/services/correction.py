"""Fix typing mistakes before translation (T32, docs/experiments.md 6).

Only typed English is corrected. In the measurement a small LLM raised en->ko chrF on text with typos from
27.0 to 34.6 for 0.3 on clean text; for Korean input every candidate cost clean text more than the rule
allowed. Correction is a helper step: when Ollama is down, slow or answers oddly, the text is translated
as typed, as before, and the server never waits for Ollama to start (T37).
"""

# [파일 역할] 글자로 입력한 영어의 오타·띄어쓰기를 번역 전에 Ollama의 작은 LLM(qwen2.5 1.5B)으로 고치는
# 교정기(TypoCorrector 구현).
# [설계 이유] opus-mt는 오타에 약해 오타를 섞은 영→한 번역 품질이 크게 떨어졌고, 교정을 거치니 validation
# chrF가 27.0 → 34.6으로 올랐다(위 docstring, docs/experiments.md 6절). 한국어는 교정이 깨끗한 문장을 망가뜨려
# 쓰지 않는다. 교정은 보조 단계라 Ollama가 없거나 느리거나 이상하게 답하면 입력 그대로 번역하고, 서버 시작도
# Ollama를 기다리지 않는다.
# [흐름] start()가 백그라운드 스레드에서 prepare()(모델 내려받기 → 적재 → 첫 교정)를 성공할 때까지 되풀이 →
# 준비되면 corrects()가 True → 요청마다 correct()가 /api/chat을 한 번 부르고, 답이 교정처럼 보일 때만 쓴다.
# [연결] services/models.load_typo_correction이 만들고, services/pipeline.prepare가
# inference.run_correction으로 correct를 부른다. app/main.py가 종료 때 close()를 부른다.

import difflib
import logging
import re
import threading

# httpx: Ollama HTTP API를 부르는 동기 클라이언트. 교정은 전용 스레드에서 돌므로 동기 호출이면 충분하다.
# threading: 백그라운드 준비 스레드와 준비 완료·종료 신호(Event), 종료와 준비 끝의 순서를 맞추는 Lock.
import httpx

from app.services.errors import describe
from app.services.interfaces import Language

logger = logging.getLogger(__name__)

# 지시문에 넣을 언어 이름. SYSTEM은 측정에 쓴 지시문 그대로이고, eval/typo_eval.py와 함께 바꿔야
# 한다(tests/test_correction.py가 둘이 같은지 확인).
LANGUAGE_NAMES: dict[Language, str] = {"ko": "Korean", "en": "English"}
# The instruction measured in docs/experiments.md 6 (eval/typo_eval.py); change both together.
SYSTEM = (
    "You fix typing mistakes. Correct only the spelling, typos and spacing in the user's {language} text. "
    "Do not change the meaning, do not translate and do not add anything. Reply with the corrected text only."
)
# The safeguard for replies that are not a correction (T38, docs/experiments.md 6). On the saved validation
# and test corrections it rejected no real correction, and caught one "The text is correct as it stands.".
# 교정이 아닌 답을 거르는 안전장치(docs/experiments.md 6절): 영어 입력의 답에 한글(자모·완성형)이 섞였으면
# 번역해 버린 것으로 본다. MIN_SIMILARITY 0.5: 입력과 답의 글자 유사도가 이보다 낮으면 거절문·설명·다른
# 문장으로 본다(docs/experiments.md 6절).
HANGUL = re.compile("[ㄱ-ㆎ가-힣]")
MIN_SIMILARITY = 0.5


# 답이 입력의 교정처럼 보이는지 판단한다. 오타 교정은 글자 대부분을 그대로 두므로 SequenceMatcher의 ratio(같은
# 글자 비율, 0~1)가 높다. autojunk=False의 이유는 아래 docstring: 기본값에서는 200자 넘는 문장의 정상 교정 두
# 개가 0.152, 0.423으로 낮게 나와 버려졌다(docs/experiments.md 6절).
def looks_like_a_correction(text: str, reply: str, language: Language) -> bool:
    """Same language and at least half the characters kept: a refusal, an explanation or a translation fails.

    autojunk=False: by default difflib ignores frequent characters in 200+ character texts, which rated good
    corrections of long sentences as unlike.
    """
    if language == "en" and HANGUL.search(reply):
        return False
    return difflib.SequenceMatcher(None, text, reply, autojunk=False).ratio() >= MIN_SIMILARITY


# temperature 0(가장 확률 높은 토큰을 고름)이라 같은 입력에는 같은 교정이 나오도록 했다.
class OllamaCorrector:
    """An instruction-tuned LLM served by Ollama with one fixed instruction, temperature 0.

    start() prepares the model on a background thread and retries until it succeeds; until then
    corrects() is False, so requests translate the text as typed instead of waiting for Ollama.
    """

    def __init__(
        self,
        model: str,
        languages: frozenset[Language] = frozenset({"en"}),
        base_url: str = "http://localhost:11434",
        timeout_s: float = 10,
        prepare_timeout_s: float = 600,
        retry_s: float = 30,
    ) -> None:
        # model_name은 기록의 mt_model에 붙는다("... + ollama/qwen2.5:1.5b-instruct"). 기본 교정 언어는
        # 영어만. 클라이언트의 기본 제한 시간은 correct()용(CORRECTION_TIMEOUT_S, 10초)이고, 오래 걸리는 준비
        # 요청은 prepare_timeout_s를 따로 준다.
        # _ready: 준비 완료 신호, _closed: 종료 신호. 둘 다 threading.Event라 요청 스레드와 준비 스레드가
        # 안전하게 함께 읽는다.
        self.model_name = f"ollama/{model}"
        self.languages = languages
        self._model = model
        self._client = httpx.Client(base_url=base_url, timeout=timeout_s)
        self._prepare_timeout_s = prepare_timeout_s
        self._retry_s = retry_s
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._worker: threading.Thread | None = None
        # Pairs close() with the worker's end, so exactly one of them closes the client and the model never
        # turns on after shutdown (T41).
        self._lock = threading.Lock()
        self._worker_done = False

    # 지금 이 언어를 교정할지: 교정 대상 언어이고, 준비가 끝났고, 닫히지 않았을 때만. 준비 전에는 False라
    # 요청이 Ollama를 기다리지 않는다.
    def corrects(self, language: Language) -> bool:
        return language in self.languages and self._ready.is_set() and not self._closed.is_set()

    def correct(self, text: str, language: Language) -> str | None:
        """The corrected text, or None when the model gave none; the caller then keeps the original.

        The log never carries the text itself.
        """
        if self._closed.is_set():
            return None
        # /api/chat을 한 번 부른다. 연결 실패·시간 초과·HTTP 오류·JSON 형식 이상은 모두 None(입력 그대로
        # 번역). 로그에는 입력 글을 남기지 않는다.
        try:
            response = self._client.post("/api/chat", json=self._payload(text, language))
            response.raise_for_status()
            body = response.json()
            content = body["message"]["content"]
            # Anything but a normal finish ("length": cut off at num_predict) would drop part of the text.
            finished = body.get("done") is True and body.get("done_reason") == "stop"
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as exc:
            logger.warning("Typo correction failed; translating the text as typed: %s", describe(exc))
            return None
        # 정상 종료(done=true, done_reason="stop")가 아니면 답이 num_predict(256토큰)에서 잘렸을 수 있어 쓰지
        # 않는다. 빈 답도 쓰지 않는다. 마지막으로 안전장치를 통과한 답만 교정문으로 돌려준다.
        if not finished or not isinstance(content, str) or not content.strip():
            logger.warning("Typo correction gave no finished reply; translating the text as typed")
            return None
        corrected = content.strip()
        if not looks_like_a_correction(text, corrected, language):
            logger.warning(
                "Typo correction reply is not a correction of the text; translating the text as typed"
            )
            return None
        return corrected

    # 준비 스레드를 띄운다. daemon=True라 이 스레드가 남아 있어도 프로세스 종료를 막지 않는다.
    # services/models.load_typo_correction이 부른다.
    def start(self) -> threading.Thread:
        self._worker = threading.Thread(
            target=self._prepare_until_ready, name="typo-correction-prepare", daemon=True
        )
        self._worker.start()
        return self._worker

    # 종료 신호를 켜고, 준비 스레드가 없거나 이미 끝났으면 여기서 HTTP 클라이언트를 닫는다. 준비 스레드가 돌고
    # 있으면 그 스레드가 끝날 때 닫는다(_prepare_until_ready의 finally). Lock으로 둘 중 정확히 한쪽만 닫게
    # 한다.
    def close(self) -> None:
        """Turn correction off for good and stop preparing (T39, called when the app shuts down).

        Never waits: a preparation call already sent ends within its time limit, and the worker then closes
        the HTTP client; with no worker running, close() closes it here.
        """
        with self._lock:
            self._closed.set()
            if self._worker is None or self._worker_done:
                self._client.close()

    # 준비 스레드 본문: 성공할 때까지 prepare()를 되풀이한다. 실패하면 경고는 첫 번째에만 남기고(재시도마다
    # 로그가 쌓이지 않게), retry_s(30초) 동안 기다린다. 기다림을 _closed.wait로 해서 종료 신호가 오면 바로
    # 깬다.
    # 성공하면 Lock 안에서 '아직 닫히지 않았을 때만' 준비 완료로 표시해, 종료 뒤에 교정이 켜지는 일을 막는다.
    def _prepare_until_ready(self) -> None:
        failures = 0
        try:
            while not self._closed.is_set():
                try:
                    self.prepare()
                except Exception as exc:  # noqa: BLE001 - any failure leaves correction off until a later try works
                    failures += 1
                    if failures == 1 and not self._closed.is_set():  # once, not every retry
                        logger.warning(
                            "Typo correction model %s is not ready; typed text is translated as is. "
                            "Trying again every %.0f s: %s",
                            self._model,
                            self._retry_s,
                            describe(exc),
                        )
                    self._closed.wait(self._retry_s)
                    continue
                with self._lock:  # a preparation that ends after shutdown stays off
                    if not self._closed.is_set():
                        self._ready.set()
                        logger.info("Typo correction model %s is ready", self._model)
                return
        finally:
            with self._lock:
                self._worker_done = True
                if self._closed.is_set():
                    self._client.close()

    # 모델 준비 세 단계: (1) /api/show로 모델이 있는지 보고 404면 /api/pull로 내려받는다(첫 compose 시작, 약
    # 1GB) (2) 프롬프트 없는 /api/generate로 모델만 올리고 keep_alive -1로 계속 올려 둔다 (3) 교정을 한 번 해
    # 둔다. 실패하면 raise_for_status 예외가 _prepare_until_ready로 올라가 재시도된다.
    def prepare(self) -> None:
        """Pull the model if Ollama lacks it (the first compose start), load it to stay, and correct once.

        Every call has a time limit (prepare_timeout_s for the slow ones), so a stalled Ollama cannot hold
        the thread forever. By default Ollama unloads a model after five idle minutes and the next request
        waits for it to load again; keep_alive -1 keeps it until Ollama stops (`ollama stop <model>`).
        """
        slow = self._prepare_timeout_s
        shown = self._client.post("/api/show", json={"model": self._model})
        if shown.status_code == 404:
            logger.info("Pulling %s into Ollama (about 1 GB, once)", self._model)
            pulled = self._client.post(
                "/api/pull", json={"model": self._model, "stream": False}, timeout=slow
            )
            pulled.raise_for_status()
        else:
            shown.raise_for_status()
        # A request without a prompt only loads the model.
        loaded = self._client.post(
            "/api/generate", json={"model": self._model, "keep_alive": -1}, timeout=slow
        )
        loaded.raise_for_status()
        # Loading alone was not enough: in the first compose start the first three corrections each ran past
        # the 10 s timeout, and later ones took 0.1-0.2 s (T32, cause not found). Pay for it here.
        first = self._client.post("/api/chat", json=self._payload("Hello.", "en"), timeout=slow)
        first.raise_for_status()

    # /api/chat 요청 본문: 시스템 지시문 + 사용자 글, 스트리밍 없이 한 번에 받는다. keep_alive -1로 요청
    # 뒤에도 모델을 올려 둔다. num_predict 256은 답의 최대 토큰 수이고, 이를 넘어 잘린 답은 correct()의
    # done_reason 검사가 버린다.
    def _payload(self, text: str, language: Language) -> dict:
        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM.format(language=LANGUAGE_NAMES[language])},
                {"role": "user", "content": text},
            ],
            "stream": False,
            "keep_alive": -1,
            "options": {"temperature": 0, "num_predict": 256},
        }

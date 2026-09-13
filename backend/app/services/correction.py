"""Fix typing mistakes before translation (T32, docs/experiments.md 6).

Only typed English is corrected. In the measurement a small LLM raised en->ko chrF on text with typos from
27.0 to 34.6 for 0.3 on clean text; for Korean input every candidate cost clean text more than the rule
allowed. Correction is a helper step: when Ollama is down, slow or answers oddly, the text is translated
as typed, as before, and the server never waits for Ollama to start (T37).
"""

import difflib
import logging
import re
import threading

import httpx

from app.services.interfaces import Language

logger = logging.getLogger(__name__)

LANGUAGE_NAMES: dict[Language, str] = {"ko": "Korean", "en": "English"}
# The instruction measured in docs/experiments.md 6 (eval/typo_eval.py); change both together.
SYSTEM = (
    "You fix typing mistakes. Correct only the spelling, typos and spacing in the user's {language} text. "
    "Do not change the meaning, do not translate and do not add anything. Reply with the corrected text only."
)
# The safeguard for replies that are not a correction (T38, docs/experiments.md 6). On the saved validation
# and test corrections it rejected no real correction, and caught one "The text is correct as it stands.".
HANGUL = re.compile("[ㄱ-ㆎ가-힣]")
MIN_SIMILARITY = 0.5


def looks_like_a_correction(text: str, reply: str, language: Language) -> bool:
    """Same language and at least half the characters kept: a refusal, an explanation or a translation fails.

    autojunk=False: by default difflib ignores frequent characters in 200+ character texts, which rated good
    corrections of long sentences as unlike.
    """
    if language == "en" and HANGUL.search(reply):
        return False
    return difflib.SequenceMatcher(None, text, reply, autojunk=False).ratio() >= MIN_SIMILARITY


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
        self.model_name = f"ollama/{model}"
        self.languages = languages
        self._model = model
        self._client = httpx.Client(base_url=base_url, timeout=timeout_s)
        self._prepare_timeout_s = prepare_timeout_s
        self._retry_s = retry_s
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._worker: threading.Thread | None = None

    def corrects(self, language: Language) -> bool:
        return language in self.languages and self._ready.is_set() and not self._closed.is_set()

    def correct(self, text: str, language: Language) -> str | None:
        """The corrected text, or None when the model gave none; the caller then keeps the original.

        The log never carries the text itself.
        """
        if self._closed.is_set():
            return None
        try:
            response = self._client.post("/api/chat", json=self._payload(text, language))
            response.raise_for_status()
            body = response.json()
            content = body["message"]["content"]
            # Anything but a normal finish ("length": cut off at num_predict) would drop part of the text.
            finished = body.get("done") is True and body.get("done_reason") == "stop"
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError):
            logger.warning("Typo correction failed; translating the text as typed", exc_info=True)
            return None
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

    def start(self) -> threading.Thread:
        self._worker = threading.Thread(
            target=self._prepare_until_ready, name="typo-correction-prepare", daemon=True
        )
        self._worker.start()
        return self._worker

    def close(self) -> None:
        """Turn correction off for good and stop preparing (T39, called when the app shuts down).

        Never waits: a preparation call already sent ends within its time limit, and the worker then closes
        the HTTP client; with no worker running, close() closes it here.
        """
        self._closed.set()
        if self._worker is None or not self._worker.is_alive():
            self._client.close()

    def _prepare_until_ready(self) -> None:
        failures = 0
        try:
            while not self._closed.is_set():
                try:
                    self.prepare()
                except Exception:  # noqa: BLE001 - any failure leaves correction off until a later try works
                    failures += 1
                    if failures == 1 and not self._closed.is_set():  # once, not every retry
                        logger.warning(
                            "Typo correction model %s is not ready; typed text is translated as is. "
                            "Trying again every %.0f s",
                            self._model,
                            self._retry_s,
                            exc_info=True,
                        )
                    self._closed.wait(self._retry_s)
                    continue
                if not self._closed.is_set():  # a preparation that ends after shutdown stays off
                    self._ready.set()
                    logger.info("Typo correction model %s is ready", self._model)
                return
        finally:
            if self._closed.is_set():
                self._client.close()

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

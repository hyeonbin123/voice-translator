"""Fix typing mistakes before translation (T32, docs/experiments.md 6).

Only typed English is corrected. In the measurement a small LLM raised en->ko chrF on text with typos from
27.0 to 34.6 for 0.3 on clean text; for Korean input every candidate cost clean text more than the rule
allowed. Correction is a helper step: when Ollama is down, slow or answers oddly, the text is translated
as typed, as before.
"""

import logging

import httpx

from app.services.interfaces import Language

logger = logging.getLogger(__name__)

LANGUAGE_NAMES: dict[Language, str] = {"ko": "Korean", "en": "English"}
# The instruction measured in docs/experiments.md 6 (eval/typo_eval.py); change both together.
SYSTEM = (
    "You fix typing mistakes. Correct only the spelling, typos and spacing in the user's {language} text. "
    "Do not change the meaning, do not translate and do not add anything. Reply with the corrected text only."
)


class OllamaCorrector:
    """An instruction-tuned LLM served by Ollama with one fixed instruction, temperature 0."""

    def __init__(
        self,
        model: str,
        languages: frozenset[Language] = frozenset({"en"}),
        base_url: str = "http://localhost:11434",
        timeout_s: float = 10,
    ) -> None:
        self.model_name = f"ollama/{model}"
        self.languages = languages
        self._model = model
        self._client = httpx.Client(base_url=base_url, timeout=timeout_s)

    def corrects(self, language: Language) -> bool:
        return language in self.languages

    def correct(self, text: str, language: Language) -> str | None:
        """The corrected text, or None when the model gave none; the caller then keeps the original.

        The log never carries the text itself.
        """
        try:
            response = self._client.post("/api/chat", json=self._payload(text, language))
            response.raise_for_status()
            body = response.json()
            content = body["message"]["content"]
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            logger.warning("Typo correction failed; translating the text as typed", exc_info=True)
            return None
        # "length" means the reply hit num_predict: a cut-off correction would drop the end of the text.
        if body.get("done_reason") == "length" or not isinstance(content, str) or not content.strip():
            logger.warning("Typo correction gave no usable reply; translating the text as typed")
            return None
        return content.strip()

    def prepare(self) -> None:
        """Pull the model if Ollama lacks it (the first compose start), then load it to stay loaded.

        By default Ollama unloads a model after five idle minutes, and the next request waits for it to load
        again; keep_alive -1 keeps it until Ollama stops (`ollama stop <model>` frees it on a desktop).
        """
        shown = self._client.post("/api/show", json={"model": self._model})
        if shown.status_code == 404:
            logger.info("Pulling %s into Ollama (about 1 GB, once)", self._model)
            pulled = self._client.post(
                "/api/pull", json={"model": self._model, "stream": False}, timeout=None
            )
            pulled.raise_for_status()
        else:
            shown.raise_for_status()
        # A request without a prompt only loads the model.
        loaded = self._client.post(
            "/api/generate", json={"model": self._model, "keep_alive": -1}, timeout=None
        )
        loaded.raise_for_status()
        # Loading alone was not enough: in the first compose start the first three corrections each ran past
        # the 10 s timeout, and later ones took 0.1-0.2 s (T32, cause not found). Pay for it here, untimed.
        first = self._client.post("/api/chat", json=self._payload("Hello.", "en"), timeout=None)
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

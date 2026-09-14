"""Live subtitles while a person speaks (T34, docs/experiments.md 8).

Each update recognizes the whole utterance so far again, and its result replaces the text on screen. The
start that two results in a row agree on is shown dark, the rest faded. Results are compared by their
letters only: case, spaces and punctuation change between results (Korean spacing above all) without
changing what was said.
"""

import io
import re
import unicodedata
import wave

_WORD = re.compile(r"\S+")
SAMPLE_RATE = 16_000


def wav_from_pcm(pcm: bytes) -> bytes:
    """16 kHz mono 16-bit PCM as the WAV file conversation mode uploads (frontend pcm.ts)."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(pcm[: len(pcm) // 2 * 2])
    return buffer.getvalue()


def letters(text: str) -> str:
    """The text without case, spaces or punctuation."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    return "".join(ch for ch in folded if not ch.isspace() and not unicodedata.category(ch).startswith("P"))


def common_start(a: str, b: str) -> int:
    count = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        count += 1
    return count


def stable_length(previous: str | None, current: str) -> int:
    """How many characters at the start of `current` to show dark: the whole words whose letters the
    previous result also starts with. Nothing is dark until there is a previous result."""
    if previous is None:
        return 0
    agreed = common_start(letters(previous), letters(current))
    stable = seen = 0
    for word in _WORD.finditer(current):
        seen += len(letters(word.group()))
        if seen > agreed:
            break
        stable = word.end()
    return stable

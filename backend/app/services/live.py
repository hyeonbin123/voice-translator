"""Live subtitles while a person speaks (T34, docs/experiments.md 8).

Each update recognizes the whole utterance so far again, and its result replaces the text on screen. The
start that two results in a row agree on is shown dark, the rest faded. Results are compared by their
letters only: case, spaces and punctuation change between results (Korean spacing above all) without
changing what was said.
"""

# [파일 역할] 동시통역 자막에 쓰는 순수 함수 모음: 브라우저가 보낸 PCM을 WAV로 감싸기, 비교용 글자만 남기기,
# 이웃한 두 인식 결과가 같은 앞부분 길이(진하게 보일 부분) 계산.
# [설계 이유] 두 결과를 그대로 비교하면 대소문자·띄어쓰기(특히 한국어)·문장부호만 바뀌어도 '다르다'가 된다.
# 말한 내용이 같으면 같다고 보도록 글자만 비교하고, 화면에는 단어 경계까지만 진하게 보여 단어 중간에서 끊기지
# 않게 한다(docs/api.md source 메시지의 stable).
# [연결] routers/live.py가 쓰고, 측정 도구 eval/stream_eval.py도 같은 함수로 오프라인 측정을 해 서비스와
# 측정이 같은 규칙을 쓴다. 테스트: tests/test_live.py.

import io
import re
import unicodedata
import wave

# _WORD: 공백으로 나뉜 단어 하나. SAMPLE_RATE: 브라우저가 보내는 PCM의 표본 추출률(16kHz, Whisper 입력 규격).
_WORD = re.compile(r"\S+")
SAMPLE_RATE = 16_000


# 웹소켓으로 받은 PCM 바이트에 WAV 머리를 붙인다. 대화 모드가 HTTP로 올리는 파일과 같은 형식이라, 최종 처리
# 결과가 업로드했을 때와 같아진다(docs/api.md: 최종 결과는 음성 번역 API와 같은 WAV로 처리).
# 16비트 표본은 2바이트이므로 홀수 길이면 마지막 1바이트를 버려 반쪽 표본이 생기지 않게 한다.
def wav_from_pcm(pcm: bytes) -> bytes:
    """16 kHz mono 16-bit PCM as the WAV file conversation mode uploads (frontend pcm.ts)."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(pcm[: len(pcm) // 2 * 2])
    return buffer.getvalue()


# 비교용 정규화: NFKC로 같은 글자의 다른 표기(전각·호환 문자 등)를 맞추고, casefold로 대소문자를 없애고,
# 공백과 문장부호(P 범주)를 뺀다. 예: "Soon, officers entered" → "soonofficersentered".
def letters(text: str) -> str:
    """The text without case, spaces or punctuation."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    return "".join(ch for ch in folded if not ch.isspace() and not unicodedata.category(ch).startswith("P"))


# 두 문자열이 앞에서부터 몇 글자까지 같은지.
def common_start(a: str, b: str) -> int:
    count = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        count += 1
    return count


# 직전 결과와 지금 결과가 '글자 기준으로' 같은 앞부분을 구한 뒤, 지금 결과(원래 문자열)에서 그 안에 완전히
# 들어가는 단어까지의 길이를 돌려준다. 단어별 글자 수를 누적(seen)하다 합의한 글자 수(agreed)를 넘는 단어에서
# 멈춘다. 반환값은 current의 문자 위치라 화면이 current[:stable]을 진하게 그린다.
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

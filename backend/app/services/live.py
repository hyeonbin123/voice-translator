"""사람이 말하는 동안 보여 주는 동시통역 자막(docs/experiments.md 8절).

갱신마다 지금까지의 마디 전체를 다시 인식하고, 그 결과가 화면의 글을 바꾼다. 연이은 두 결과가 같은
앞부분은 진하게, 나머지는 흐리게 보인다. 결과는 글자만으로 비교한다: 대소문자·띄어쓰기·문장부호는 말한
내용이 같아도 결과마다 바뀌기 때문이다(무엇보다 한국어 띄어쓰기).
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


def wav_from_pcm(pcm: bytes) -> bytes:
    """16kHz 모노 16비트 PCM을 대화 모드가 올리는 것과 같은 WAV 파일로 만든다(frontend pcm.ts).

    웹소켓으로 받은 PCM 바이트에 WAV 머리를 붙인다. HTTP로 올리는 파일과 같은 형식이라 최종 처리 결과가
    업로드했을 때와 같아진다(docs/api.md: 최종 결과는 음성 번역 API와 같은 WAV로 처리). 16비트 표본은
    2바이트이므로 홀수 길이면 마지막 1바이트를 버려 반쪽 표본이 생기지 않게 한다.
    """
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(pcm[: len(pcm) // 2 * 2])
    return buffer.getvalue()


def letters(text: str) -> str:
    """대소문자·공백·문장부호를 뺀 글자만 남긴다.

    비교용 정규화: NFKC로 같은 글자의 다른 표기(전각·호환 문자 등)를 맞추고, casefold로 대소문자를
    없애고, 공백과 문장부호(P 범주)를 뺀다. 예: "Soon, officers entered" → "soonofficersentered".
    """
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


def stable_length(previous: str | None, current: str) -> int:
    """`current` 앞에서 몇 글자를 진하게 보일지 돌려준다.

    진하게 보이는 것은 직전 결과도 같은 글자로 시작하는 완전한 단어들이다. 직전 결과가 없으면 아무것도
    진하게 보이지 않는다. 직전 결과와 지금 결과가 '글자 기준으로' 같은 앞부분(agreed)을 구한 뒤, 지금
    결과(원래 문자열)의 단어별 글자 수를 누적(seen)하다 agreed를 넘는 단어에서 멈춘다. 반환값은 current의
    문자 위치라 화면이 current[:stable]을 진하게 그린다.
    """
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

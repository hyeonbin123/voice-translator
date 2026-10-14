"""두 사람이 한 화면으로 대화할 때 한 마디가 어느 언어인지 정한다(docs/experiments.md 10절).

HTTP 두 사람 대화 API와 동시통역 웹소켓의 두 사람 대화 모드가 모두 이 규칙으로 정한다.
"""

# [파일 역할] 두 사람 대화 모드에서 한 마디가 어느 언어인지 최종 결정하는 규칙 하나를 둔다.
# [설계 이유] Whisper의 언어 판별이 확신이 낮을 때(짧은 대답 등)는 '두 사람이 번갈아 말한다'는 대화의 성질을
# 믿는 편이 낫다고 보고, 직전 마디의 반대 언어로 처리한다. 문턱 0.6은 validation에서
# 골랐다(docs/experiments.md 10절, DIALOG_LANGUAGE_THRESHOLD). HTTP와 웹소켓이 같은 규칙을 쓰도록 한 함수로
# 뺐다.
# [연결] routers/translate.py(POST /api/translate/dialog)와 routers/live.py(mode "dialog")가 부른다.

from app.services.interfaces import Language

# 각 언어의 상대 언어. 판별한 언어에서 번역할 언어를 정하는 데도 쓴다.
OTHER: dict[Language, Language] = {"ko": "en", "en": "ko"}


def choose_language(
    detected: Language, confidence: float, previous: Language | None, threshold: float
) -> tuple[Language, bool]:
    """한 마디의 언어와 그것을 추측했는지를 돌려준다.

    판별이 애매하고(확신이 문턱보다 낮고) 앞 마디가 알려져 있으면 대화가 번갈아 이어진다고 보고, 이 마디를
    앞 마디와 반대 언어로 본다.
    입력: 판별한 언어, 그 확신(0.5~1.0), 직전 마디의 언어(첫 마디면 None), 문턱. 출력: (처리할 언어,
    추측했는지). 첫 마디는 비교할 직전 마디가 없으므로 확신이 낮아도 판별 결과를 그대로 쓴다.
    추측(True)이면 화면이 방향 바꾸기 단추를 보여 준다.
    """
    if previous is not None and confidence < threshold:
        return OTHER[previous], True
    return detected, False

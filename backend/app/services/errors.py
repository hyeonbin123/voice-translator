"""실패를 그 뒤에 있는 예외들의 메시지 없이 로그용으로 설명한다.

모델 라이브러리의 메시지에는 입력받은 글이 그대로 들어갈 수 있고, traceback은 원인 사슬 전체를 찍는다.
그래서 로그에는 사슬을 따라 예외 종류와 가장 안쪽 예외가 난 위치만 남긴다. 이것만으로도 어느 단계가 왜
실패했는지 알 수 있다.
"""

# [파일 역할] 예외를 로그에 남길 때 메시지 없이 '예외 종류의 사슬 + 가장 안쪽 예외가 난 위치'만으로 요약하는
# 함수 하나를 둔다.
# [설계 이유] 모델 라이브러리의 예외 메시지에는 사용자가 입력한 글이나 번역문이 그대로 들어갈 수 있다. 그런
# 개인 정보가 서버 로그에 남지 않게 하면서도, 어느 단계가 왜 실패했는지는 알 수 있게 한다(위 docstring).
# [연결] app/main.py의 로그 필터(WithoutExceptionMessages), routers/health.py, routers/live.py,
# routers/translate.py, services/pipeline.py, services/correction.py가 로그를 쓸 때 부른다.

import traceback
from pathlib import Path


def describe(exc: BaseException) -> str:
    """예: "ModelError <- ValueError (at sentencepiece.py:123 in encode)". 메시지는 절대 넣지 않는다.

    입력은 예외 객체, 출력은 "바깥 예외 <- 원인 예외 ... (at 파일:줄 in 함수)" 형태의 한 줄 문자열이다.
    """
    names: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    innermost = exc
    # 원인 사슬을 따라 안쪽으로 내려간다. `raise ... from`으로 단 원인(__cause__)을 먼저 보고, 없으면 처리
    # 중에 생긴 예외(__context__)를 본다. 단 `from None`으로 숨긴 문맥(__suppress_context__)은 따라가지
    # 않는다. seen은 서로를 가리키는 사슬에서 무한 반복을 막는다.
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        names.append(type(current).__name__)
        innermost = current
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
    # 가장 안쪽 예외의 traceback에서 마지막 프레임(실제로 예외가 난 곳)의 파일 이름(경로 제외)·줄·함수만
    # 붙인다. traceback이 없으면(직접 만든 예외 객체) 위치는 생략한다.
    frames = traceback.extract_tb(innermost.__traceback__)
    where = (
        f" (at {Path(frames[-1].filename).name}:{frames[-1].lineno} in {frames[-1].name})" if frames else ""
    )
    return " <- ".join(names) + where

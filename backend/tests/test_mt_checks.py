# [파일 역할] LLM 번역기(Hy-MT2) 시험의 출력 점검 eval/mt_checks.py를 검증한다(docs/experiments.md 15절 판단
# 규칙 (b)).
# [점검 내용] 번역문이 (1) 목표 언어가 쓰지 않고 입력에도 없는 문자(한자·가나·키릴 등)를 담았는지, (2)
# 설명하는 티(앞머리 꼬리표, 주석, 입력보다 많은 줄, 입력에 없던 따옴표, 참조의 두 배를 넘는 길이)가 나는지
# 센다. 빈 출력은 따로 센다.
# [왜 필요한가] 15절 판단 규칙 (b)는 validation 129문장 중 다른 문자·설명형 출력이 1개 이하이고 빈 출력이
# 0개일 것을 요구한다. 이 함수들이 그 개수를 세므로, 점검 규칙이 바뀌면 기록된 판정이 달라진다. 그래서 측정
# 전에 정한 규칙을 사례로 고정한다.

import pytest

from eval.mt_checks import check_items, explanation_signs, foreign_letters

# 설명형 사례에 쓰는 참조 번역. 출력이 참조의 두 배를 넘으면 'long'이 붙으므로, 꼬리표·주석 사례가 길이 때문에
# 'long'까지 걸리지 않을 만큼 길게 둔다.
REF = "서울은 대한민국의 수도이고 가장 큰 도시다."  # long enough that the label cases are not 'long'


# 다른 문자 점검: 한국어 출력의 라틴 문자·숫자(NASA, 2026)는 허용하고, 한자·가타카나·키릴 문자는 잡는다.
# 입력에 이미 있는 글자(그리스 문자 α, 한국어 원문의 한자)는 출력에 남아도 허용한다. 단 영어 출력에 남은
# 한글은 입력이 한국어여도 번역하지 않고 남긴 것이므로 늘 잡는다. 라틴 확장 문자(é)는 라틴으로 허용되고,
# 기호(€)는 글자가 아니라 보지 않는다.
@pytest.mark.parametrize(
    ("output", "source", "target", "expected"),
    [
        ("서울은 한국의 수도이다.", "Seoul is the capital of Korea.", "ko", ""),
        ("NASA는 2026년에 발표했다.", "NASA announced it in 2026.", "ko", ""),
        ("서울은 韓國의 수도이다.", "Seoul is the capital of Korea.", "ko", "韓國"),
        ("東京은 일본의 수도다.", "Tokyo is the capital of Japan.", "ko", "東京"),
        ("카타카나 カナ 섞임", "Katakana mixed in", "ko", "カナ"),
        ("러시아어 слово 섞임", "Russian word mixed in", "ko", "слово"),
        # A character the input itself has is kept: a Greek letter, or a Hanja in a Korean source.
        ("The α particle decays.", "α 입자가 붕괴한다.", "en", ""),
        ("Hanja 漢 stays.", "한자 漢이 남는다.", "en", ""),
        # Korean left in an English output always counts, even though the input is Korean.
        ("He went to 학교.", "그는 학교에 갔다.", "en", "학교"),
        ("Café au lait costs 3 €.", "카페오레는 3유로다.", "en", ""),
    ],
)
def test_foreign_letters(output, source, target, expected):
    assert foreign_letters(output, source, target) == expected


# 설명형 점검: 'Translation:'·'번역:'·'Here is the translation' 같은 꼬리표(label), '(Note: ...)'·'※' 같은
# 주석(note), 입력보다 많은 줄(lines), 입력에 없던 전체 따옴표(quotes), 참조 글자 수의 두 배 초과(long,
# 반복이나 덧붙인 글)를 각각 알아보는지 확인한다.
# 입력이 이미 따옴표로 감싸여 있거나 '참고:'로 시작하면 출력의 같은 표시는 입력을 옮긴 것이라 세지 않는다.
@pytest.mark.parametrize(
    ("output", "source", "reference", "expected"),
    [
        ("서울은 수도이다.", "Seoul is the capital.", "서울은 수도다.", []),
        ("Translation: 서울은 수도이다.", "Seoul is the capital.", REF, ["label"]),
        ("번역: 서울은 수도이다.", "Seoul is the capital.", REF, ["label"]),
        ("Here is the translation: Seoul.", "서울.", "Seoul is a big city.", ["label"]),
        ("서울은 수도이다. (Note: literal)", "Seoul is the capital.", REF, ["note"]),
        ("서울은 수도이다. ※ 직역", "Seoul is the capital.", "서울은 수도다.", ["note"]),
        ("서울은 수도이다.\n참고로 서울은 크다.", "Seoul is the capital.", REF, ["lines"]),
        ('"서울은 수도이다."', "Seoul is the capital.", "서울은 수도다.", ["quotes"]),
        ('"서울은 수도이다."', '"Seoul is the capital."', "서울은 수도다.", []),
        (
            "서울은 수도이다. 서울은 수도이다. 서울은 수도이다.",
            "Seoul is the capital.",
            "서울은 수도다.",
            ["long"],
        ),
        # A note marker the input already has is the input's, not an added note.
        ("참고: 서울은 수도이다.", "Note: Seoul is the capital.", "참고: 서울은 수도다.", []),
    ],
)
def test_explanation_signs(output, source, reference, expected):
    assert explanation_signs(output, source, reference) == expected


# 평가 항목 묶음 집계: 다른 문자(id 2)와 설명형(id 3)이 각각 1개, 둘 중 하나라도 걸린 것(flagged) 2개이고, 빈
# 출력(id 4)은 걸린 것에 넣지 않고 따로(empty) 센다. 빈 출력은 번역기가 예외를 던져 평가가 ""를 저장한 경우라
# 출력 품질과 다른 문제이기 때문이다(모듈 docstring).
def test_check_items_counts_flagged_and_empty_outputs_apart():
    items = [
        {"id": 1, "source": "Seoul.", "reference": "서울.", "hypothesis": "서울."},
        {"id": 2, "source": "Tokyo.", "reference": "도쿄.", "hypothesis": "東京."},
        {"id": 3, "source": "Busan.", "reference": "부산.", "hypothesis": "Translation: 부산."},
        {"id": 4, "source": "Daegu.", "reference": "대구.", "hypothesis": ""},
    ]
    result = check_items(items, "en-ko")
    assert (result["count"], result["flagged"], result["foreign_script"], result["explanatory"]) == (
        4,
        2,
        1,
        1,
    )
    assert result["empty"] == 1 and result["empty_ids"] == [4]
    assert [f["id"] for f in result["flagged_items"]] == [2, 3]

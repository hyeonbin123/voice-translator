import pytest

from eval.mt_checks import check_items, explanation_signs, foreign_letters

REF = "서울은 대한민국의 수도이고 가장 큰 도시다."  # long enough that the label cases are not 'long'


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

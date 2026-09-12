from eval.text_norm import normalize


def test_korean_drops_punctuation_and_every_space():
    assert normalize("다리 밑 수직 간격은, 15미터이며.", "ko") == "다리밑수직간격은15미터이며"
    assert normalize("할 수 있다", "ko") == normalize("할수 있다", "ko")


def test_english_lowercases_and_drops_punctuation_but_keeps_apostrophes():
    assert normalize("However, it's 25 to 30 years.", "en") == "however it's 25 to 30 years"
    assert normalize("  Well-known   ", "en") == "well known"

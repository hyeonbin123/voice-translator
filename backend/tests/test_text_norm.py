# [파일 역할] 음성 인식 오류율(한국어 CER, 영어 WER)을 재기 전에 정답과 인식 결과를 같은 규칙으로 맞추는
# eval/text_norm.py의 normalize를 검증한다(규칙은 docs/experiments.md).
# [왜 중요한가] 정규화 규칙이 바뀌면 README에 적힌 CER·WER 수치가 같은 출력으로도 달라진다.
# 띄어쓰기·문장부호처럼 말한 내용과 무관한 차이를 오류로 세지 않는 규칙을 고정한다.

from eval.text_norm import normalize


# 한국어는 문장부호와 모든 공백을 지운다. 한국어 CER은 공백 없는 글자로 재므로 '할 수'와 '할수'의 띄어쓰기
# 차이가 오류로 세어지지 않는다.
def test_korean_drops_punctuation_and_every_space():
    assert normalize("다리 밑 수직 간격은, 15미터이며.", "ko") == "다리밑수직간격은15미터이며"
    assert normalize("할 수 있다", "ko") == normalize("할수 있다", "ko")


# 한국어의 작은따옴표는 인용 부호이므로 문장부호처럼 지운다(커밋 c692191: 영어 축약형 때문에 남기던
# 아포스트로피가 한국어에서도 남던 문제를 고침).
# 두 번째 확인은 멱등성이다. 저장된 보고서에는 이미 정규화된 글자가 있으므로, 다시 정규화하면 이 수정만
# 더해지고 나머지는 그대로여야 모델을 다시 돌리지 않고 다시 채점할 수 있다.
def test_korean_drops_apostrophes_used_as_quotation_marks():
    assert normalize("'희망의 섬'이라는 이름", "ko") == "희망의섬이라는이름"
    # Saved reports hold normalized text; normalizing again only applies this fix.
    assert normalize(normalize("'희망'", "ko"), "ko") == "희망"


# 영어는 소문자로 바꾸고 문장부호를 지우되 축약형의 아포스트로피(it's)는 남기며, WER을 단어 단위로 재므로
# 공백은 한 칸씩 남긴다. 하이픈은 공백으로 바뀌어 'Well-known'이 두 단어가 된다.
def test_english_lowercases_and_drops_punctuation_but_keeps_apostrophes():
    assert normalize("However, it's 25 to 30 years.", "en") == "however it's 25 to 30 years"
    assert normalize("  Well-known   ", "en") == "well known"

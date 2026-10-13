"""Text normalization for speech recognition error rates (rules in docs/experiments.md)."""

# [파일 역할] 음성 인식 오류율(한국어 CER, 영어 WER)을 계산하기 전에 정답 전사와 인식 결과를 같은 규칙으로
# 정규화하는 함수다.
# [왜 필요한가] 정규화가 다르면 띄어쓰기·문장부호·대소문자 차이가 오류로 세어져 모델 비교가 흐려진다. 규칙은
# 측정 전에 docs/experiments.md 1절 지표에 적었다(한국어: 문장부호와 공백을 모두 뺀 글자, 영어:
# 소문자·문장부호 제거·아포스트로피 유지·공백 정리).
# [이력] 처음에는 영어 축약형을 위해 남긴 아포스트로피가 한국어에도 남았는데, 한국어에서는 인용 부호라 규칙과
# 달랐다. 그래서 한국어만 아포스트로피를 빼도록 고쳤다(T15, 1절 '보정: 한국어 아포스트로피').
# [연결] eval/stt_eval.py, eval/stt_rescore.py, eval/quality_eval.py, eval/tts_eval.py(합성 음성 재인식)가
# 쓰고 tests/test_text_norm.py가 검증한다.

import re

# _PUNCTUATION: 글자·숫자·공백·아포스트로피가 아닌 문자, 그리고 밑줄. 정규식의 단어 문자 클래스는 한글을
# 포함하지만 밑줄도 글자로 보므로 '|_'로 밑줄을 따로 넣었다.
# Anything that isn't a letter, digit, whitespace or apostrophe. `\w` covers Hangul.
_PUNCTUATION = re.compile(r"[^\w\s']|_")


# 입력: 원문, 언어('ko'·'en'). 출력: 정규화된 문자열.
# 처리: 소문자로 바꾸고 문장부호를 공백으로 바꾼 뒤, 한국어는 아포스트로피까지 공백으로 바꾸고 모든 공백을
# 지운다(글자만 남겨 CER). 영어는 연속 공백을 하나로 정리한다(단어 단위 WER).
# 숫자 표기('이백만' 대 '2,000,000')는 정규화하지 않는다. 모든 후보에 같은 조건이라 한계로만 적었다(1절 지표).
def normalize(text: str, language: str) -> str:
    """Lowercase and drop punctuation; Korean also drops every space and apostrophe.

    Korean CER is measured on characters without spaces, so a spacing difference
    ("할 수" vs "할수") doesn't count as an error. Apostrophes are kept only for English
    contractions ("it's"); in Korean they are quotation marks ('희망') and count as punctuation.
    English keeps single spaces for WER.
    """
    text = _PUNCTUATION.sub(" ", text.lower())
    if language == "ko":
        return "".join(text.replace("'", " ").split())
    return " ".join(text.split())

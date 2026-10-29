"""MeloTTS 한국어 전처리의 g2pkk 정규식 캐시(app/services/tts.py _cache_g2pkk_regex)가 음소를 바꾸지 않는지
실제 g2pkk로 확인한다."""

# [파일 역할] 실제 g2pkk G2p로 정규식 캐시를 켜기 전(g2pkk 모듈들의 진짜 re)과 켠 뒤(대리 객체)의 출력이
# 글자까지 같은지 본다. 입력은 CI에도 있는 것만 쓴다: 설치된 g2pkk의 rules.txt 예시(표준 발음법 조항마다 든
# 낱말·어구), idioms.txt 규칙의 왼쪽(관용 표기 바꾸기의 입력), 영어·숫자가 섞인 문장(convert_eng·convert_num
# 경로), 저장소에 커밋된 동시통역·대화 모드 측정 보고서(docs/experiments.md 17절)의 영→한 번역문 — 서비스가
# 실제로 합성한 한국어다. 번역문은 문장 통째로와, MeloTTS처럼 어절(띄어쓰기 단위)마다 따로 부른다. FLEURS
# 한국어 글은 data/에만 있어(커밋하지 않음) 파일과 pyarrow가 있을 때만 문장 통째로 더한다(테스트를 건너뛰지
# 않고 입력만 늘어남). 캐시가 실제로 컴파일을 건너뛰는지(같은 입력을 두 번째 부를 때 새로 컴파일하는 패턴 0)와
# 크기가 상한의 절반 아래에서 멈추는지, 끼운 모듈이 다섯인지도 본다.
# [왜 중요한가] 이 캐시는 서드파티 모듈의 전역 이름(re)을 바꿔 끼운다. g2pkk 판이 바뀌어 re를 쓰는 모듈·함수나
# 패턴 수가 달라지면 끼우는 자리가 어긋나거나 출력이 달라질 수 있는데, 그런 변화를 이 테스트가 잡는다.
# [실행] g2pkk·python-mecab-ko는 tts 의존성 그룹에만 있어 기본 pytest에서는 이 파일 전체를 건너뛴다. CI는 이
# 파일만 uv.lock과 같은 판의 g2pkk와 그 의존성을 얹어 돌린다(.github/workflows/ci.yml). g2pkk는 import할 때
# NLTK cmudict가 없으면 내려받는다(g2pkk/g2pkk.py 머리). 캐시를 끈 쪽은 호출마다 정규식 약 570개를 다시
# 컴파일해 한 번에 수십 ms가 걸려, 이 파일은 수십 초 걸린다.

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

# importorskip이 g2pkk·형태소 분석기가 없을 때 이 파일을 건너뛰게 하므로, 그 뒤의 import는 이 줄 아래에 있어야
# 한다(E402 경고를 끄는 표시는 그래서 붙어 있다). mecab(python-mecab-ko)이 없으면 G2p()가 subprocess로 pip
# install을 띄우므로 G2p를 만들기 전에 반드시 확인한다.
pytest.importorskip("g2pkk")
mecab = pytest.importorskip("mecab")
import g2pkk.g2pkk  # noqa: E402

from app.services import tts  # noqa: E402

# 캐시가 끼워져야 하는 g2pkk 0.1.2의 모듈(re를 쓰는 모든 모듈. regular.py에는 re가 없다).
EXPECTED_MODULES = ("g2pkk.g2pkk", "g2pkk.special", "g2pkk.utils", "g2pkk.english", "g2pkk.numerals")
G2PKK_DIR = Path(g2pkk.g2pkk.__file__).resolve().parent
BACKEND = Path(__file__).resolve().parents[1]
# 영어 원문 40마디의 영→한 번역문이 든, 저장소에 커밋된 측정 보고서 두 개(17절 동시통역·대화 모드 validation).
STREAM_REPORTS = (
    BACKEND / "eval" / "reports" / "stream_t95_live_dev_20261015_145547.json",
    BACKEND / "eval" / "reports" / "stream_t95_conv_dev_20261015_151242.json",
)
FLEURS_KO = BACKEND.parent / "data" / "fleurs" / "ko_kr"

# convert_eng(영어 낱말 → 한글, cmudict)과 convert_num(아라비아 숫자 → 한글, 단위 명사 앞)을 지나는 문장.
# 첫 문장은 g2pkk G2p.__call__ docstring의 예다.
MIXED = [
    "나의 친구가 mp3 file 3개를 다운받고 있다",
    "지금 mp3 파일을 다운받고 있어요",
    "오늘은 2026년 10월 29일이고 기온은 15도입니다",
    "GPU 2개로 모델을 3시간 동안 학습했어요",
    "회의는 오후 3시 30분에 online으로 합니다",
    "가격은 1,500원이고 재고는 25개 남았습니다",
    "I love Seoul 그리고 부산도 좋아요",
    "The speech model 1개와 translation model 2개",
]


# rules.txt의 '->' 줄에서 '말[발음]' 꼴의 왼쪽(입력)을 고른다. 정규식은 g2pkk utils._get_examples와 같지만, 그
# 함수는 작업 폴더 기준 'rules.txt'를 열어 여기서 그대로 쓸 수 없다. 같은 낱말은 한 번만 넣는다.
def rules_examples() -> list[str]:
    found: list[str] = []
    for line in (G2PKK_DIR / "rules.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("->"):
            found += [inp for inp, _ in re.findall(r"([ㄱ-힣][ ㄱ-힣]*)\[([ㄱ-힣][ ㄱ-힣]*)]", line)]
    return list(dict.fromkeys(found))


# idioms.txt의 규칙 줄('#' 뒤는 주석, '왼쪽===오른쪽')에서 왼쪽. G2p.idioms가 같은 방식으로 읽어 왼쪽을
# 정규식으로 바꾼다. 여기서는 그 글 자체를 입력으로 넣어 각 규칙이 실제로 걸리게 한다.
def idiom_inputs() -> list[str]:
    found: list[str] = []
    for line in (G2PKK_DIR / "idioms.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if "===" in line:
            found.append(line.split("===")[0])
    return list(dict.fromkeys(found))


# 두 측정 보고서에서 영어 원문 마디의 번역문(result.translated_text). 두 모드가 같은 클립이라 겹치는 글은 한
# 번만.
def report_translations() -> list[str]:
    found: list[str] = []
    for path in STREAM_REPORTS:
        for utterance in json.loads(path.read_text(encoding="utf-8"))["details"]["en"]:
            text = (utterance.get("result") or {}).get("translated_text")
            if text:
                found.append(text)
    return list(dict.fromkeys(found))


# data/의 FLEURS 한국어 validation·test 원문(raw_transcription). 파일이나 pyarrow가 없으면 빈 목록이다. 지표를
# 내지 않고 같은 입력의 출력이 같은지만 보므로 test 글을 써도 평가 데이터를 소모하지 않는다.
def fleurs_texts() -> list[str]:
    if importlib.util.find_spec("pyarrow") is None:
        return []
    import pyarrow.parquet as pq

    found: list[str] = []
    for split in ("validation", "test"):
        path = FLEURS_KO / f"{split}.parquet"
        if path.exists():
            found += pq.read_table(path, columns=["raw_transcription"]).column(0).to_pylist()
    return list(dict.fromkeys(found))


# 실제 G2p 하나를 이 파일 동안 함께 쓴다(cmudict 적재에 약 1초). Windows에서는 g2pkk가 eunjeon을 찾고 없으면
# pip install을 띄우므로, 서비스(app/services/tts.py _prepare_melo_on_windows)와 같이 형태소 분석기
# 확인·생성을 python-mecab-ko로 바꾼다. 바꾼 것은 이 파일이 끝나면 되돌린다.
@pytest.fixture(scope="module")
def g2p():
    with pytest.MonkeyPatch.context() as patch:
        if sys.platform == "win32":
            patch.setattr(g2pkk.g2pkk.G2p, "check_mecab", lambda self: None)
            patch.setattr(g2pkk.g2pkk.G2p, "get_mecab", lambda self: mecab.MeCab())
        yield g2pkk.g2pkk.G2p()


def test_regex_cache_keeps_every_g2pkk_output(g2p):
    rules, idioms, translations = rules_examples(), idiom_inputs(), report_translations()
    # 입력을 못 읽어 비교가 헛돌지 않게 개수를 먼저 확인한다(g2pkk 0.1.2: 예시 약 330개, 관용 규칙 약 120개,
    # 보고서 번역문 40개 안팎).
    assert len(rules) > 250 and len(idioms) > 100 and len(translations) >= 35
    words = list(dict.fromkeys(word for text in translations for word in text.split()))
    inputs = rules + idioms + MIXED + translations + words + fleurs_texts()
    assert all(sys.modules[name].re is re for name in EXPECTED_MODULES)  # 캐시가 꺼진 채로 시작한다

    before = [g2p(text) for text in inputs]
    names, restore = tts._cache_g2pkk_regex()
    try:
        assert names == EXPECTED_MODULES
        assert all(sys.modules[name].re is tts._G2PKK_RE for name in EXPECTED_MODULES)
        after = [g2p(text) for text in inputs]

        # 같은 입력을 다시 부르면 모든 패턴이 캐시에서 나온다: 적중만 늘고 새 컴파일(misses)은 0.
        sample = translations[0]
        g2p(sample)
        first = tts._G2PKK_RE.cache_info()
        g2p(sample)
        second = tts._G2PKK_RE.cache_info()
        assert second.misses == first.misses and second.hits > first.hits
        # 패턴은 g2pkk 코드·table.csv·idioms.txt의 고정 문자열이라 입력이 늘어도 크기가 늘지 않는다(약 570개).
        assert second.currsize < second.maxsize / 2
    finally:
        restore()
    assert all(sys.modules[name].re is re for name in EXPECTED_MODULES)

    changed = [(text, old, new) for text, old, new in zip(inputs, before, after, strict=True) if old != new]
    assert changed == []

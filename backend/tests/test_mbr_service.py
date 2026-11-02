"""서비스용 MBR 디코딩(app/services/mbr.py)을 번역 모델 없이 검증한다(docs/experiments.md 20절 채택 반영).

표준 라이브러리만으로 옮긴 문장 chrF가 sacrebleu 2.6.0 공개 함수·측정 도구(eval/mbr.py)와 소수점 끝자리까지
같은지, 고르기가 측정 도구와 같은 후보를 고르는지, 감싸개가 문장마다 고르고 서비스처럼 공백 하나로 잇는지
본다. sacrebleu가 필요한 테스트만 그 패키지가 없으면 건너뛴다(CI의 sacrebleu 단계가 이 파일을 다시 돈다).
"""

# [파일 역할] app/services/mbr.py의 세 부분(chrf_matrix·sentence_chrf, select, MbrTranslator)과 그것이 측정 때
# 쓴 도구(eval/mbr.py)와 같은 값·같은 후보를 내는지 본다.
# [왜 중요한가] 20절은 eval/mbr.py(sacrebleu 효용)로 잰 B8-MBR 팔을 채택했다. 서버는 sacrebleu 없이 같은
# 계산을 해야 측정한 그 디코딩을 서비스한다. 값이 끝자리까지 같지 않으면 동점 처리가 달라져 다른 후보를 고를
# 수 있다.

import json
import random
from pathlib import Path

import pytest

from app.services import mbr, translation
from app.services.interfaces import ModelError

# 서로 다른 시험 문장을 만들 단어 목록(tests/test_mbr.py와 같은 방식). 한국어 어절도 섞어 두 언어의 n-gram을
# 모두 본다.
WORDS = "the river runs past the old mill and the children wave at the boats every summer morning".split()
KOREAN = "강이 오래된 방앗간을 지나 흐르고 아이들이 여름 아침마다 배를 향해 손을 흔든다".split()
# 20절 측정 보고서(여섯 팔의 실제 번역문, 커밋됨). 실제 한국어·영어 번역문으로도 같은 값인지 본다.
MEASURED_REPORT = (
    Path(__file__).resolve().parents[1] / "eval" / "reports" / "mt_t108_dev_20261102_205405.json"
)


# ---- 문장 chrF (sacrebleu 없이 알 수 있는 값)


# sacrebleu를 쓰지 않고도 정해지는 값들: 같은 글은 100, 글자가 하나도 겹치지 않으면 0, 가설이나 참조가 비면 0,
# 공백은 세지 않으므로 띄어쓰기만 다른 글은 100, 6자보다 짧은 글은 있는 차수만으로 센다.
@pytest.mark.parametrize(
    ("hypothesis", "reference", "score"),
    [
        ("비가 와서 집에 있었다.", "비가 와서 집에 있었다.", 100.0),
        ("abc", "xyz", 0.0),
        ("", "abc", 0.0),
        ("abc", "", 0.0),
        ("", "", 0.0),
        ("a b", "ab", 100.0),
        ("ab", "ab", 100.0),
    ],
)
def test_sentence_chrf_known_values(hypothesis, reference, score):
    assert mbr.sentence_chrf(hypothesis, reference) == score


# chrF는 대칭이 아니다(beta 2라 재현율 쪽 무게가 크다): 참조를 다 담은 긴 가설이 그 반대보다 높다.
def test_sentence_chrf_weighs_recall_more_than_precision():
    assert mbr.sentence_chrf("비가 와서 집에 있었다.", "비가 와서") > mbr.sentence_chrf(
        "비가 와서", "비가 와서 집에 있었다."
    )


# 한국어·영어·문장부호·연속 공백·빈 글·같은 글·짧은 글을 모두 짝지은 표가 sacrebleu의 공개 함수
# sentence_score(2절과 같은 기본 설정)와, 측정 도구의 표(eval.mbr.chrf_matrix)와 소수점 끝자리까지 같다.
def test_chrf_matrix_matches_sacrebleu_and_the_measurement_tool():
    pytest.importorskip("sacrebleu")
    from sacrebleu.metrics import CHRF

    from eval import mbr as measured

    texts = [
        "비가 와서 집에 있었다.",
        "비가 와서 집에 머물렀다.",
        "It rained, so I stayed home.",
        "집에 있었다",
        "",
        "  띄어  쓰기 ",
        "(괄호) 안의 말!",
        "비가 와서 집에 있었다.",
        "ab",
        "The rain kept me home; it rained all day.",
    ]
    table = mbr.chrf_matrix(texts)
    assert table == measured.chrf_matrix(texts)
    metric = CHRF()
    for i, hypothesis in enumerate(texts):
        for j, reference in enumerate(texts):
            if i != j:
                assert table[i][j] == metric.sentence_score(hypothesis, [reference]).score
                assert mbr.sentence_chrf(hypothesis, reference) == table[i][j]


# 20절 측정 보고서의 실제 번역문(한→영 세 팔·영→한 세 팔, 입력마다)으로도 같은 값이다: 측정 때 같은 입력의 팔
# 출력끼리 견준 표를 서비스 구현으로 다시 계산해 측정 도구·sacrebleu와 견준다.
def test_chrf_matrix_matches_sacrebleu_on_the_measured_translations():
    pytest.importorskip("sacrebleu")
    from sacrebleu.metrics import CHRF

    from eval import mbr as measured

    results = json.loads(MEASURED_REPORT.read_text(encoding="utf-8"))
    metric = CHRF()
    compared = 0
    for direction in ("ko-en", "en-ko"):
        arms = [r["directions"][direction]["items"] for r in results if direction in r["directions"]]
        assert len(arms) == 3
        for items in zip(*arms, strict=True):
            texts = [item["hypothesis"] for item in items]
            table = mbr.chrf_matrix(texts)
            assert table == measured.chrf_matrix(texts)
            for i, hypothesis in enumerate(texts):
                for j, reference in enumerate(texts):
                    if i != j:
                        assert table[i][j] == metric.sentence_score(hypothesis, [reference]).score
                        compared += 1
    assert compared == 2 * 129 * 6


# ---- MBR 고르기


# 정해 둔 효용 표를 돌려주는 대역. 받은 후보 글 목록을 calls에 남긴다(표를 부르지 않아야 하는 경우를 확인).
def table_matrix(table):
    calls = []

    def matrix(texts):
        calls.append(list(texts))
        return table

    matrix.calls = calls
    return matrix


# 다른 후보들과 견준 평균 효용이 가장 큰 후보를 고른다(자기 자신은 빼고). 동점이면 엔진 점수, 그것도 같으면 앞
# 후보다(측정 도구와 같은 규칙).
def test_select_takes_the_highest_mean_utility_then_score_then_order():
    table = [[0, 10, 10], [50, 0, 10], [50, 10, 0]]
    assert mbr.select([("a", -0.1), ("b", -0.2), ("c", -0.3)], table_matrix(table)) == 1
    tie = [[0, 30, 30], [30, 0, 30], [30, 30, 0]]
    assert mbr.select([("a", -0.5), ("b", -0.2), ("c", -0.2)], table_matrix(tie)) == 1
    assert mbr.select([("a", -0.2), ("b", -0.2), ("c", -0.2)], table_matrix(tie)) == 0


# 후보가 하나면 표를 만들지 않고 0이고, 없으면 ValueError다.
def test_select_single_or_no_candidate():
    matrix = table_matrix([[0]])
    assert mbr.select([("only", -1.0)], matrix) == 0
    assert matrix.calls == []
    with pytest.raises(ValueError):
        mbr.select([], matrix)


# 실제 chrF로: 엔진 1-best(점수 −0.1)가 다른 후보들과 동떨어진 글이면 서로 가까운 후보들 쪽을 고르고, 겹치는
# 글은 따로 세어 같은 글을 낸 가설이 많은 쪽이 유리하다.
def test_select_prefers_the_consensus_with_the_real_chrf():
    candidates = [
        ("a dog barked loudly at night.", -0.1),
        ("the cat sat on the mat.", -0.3),
        ("the cat sat on the mat.", -0.4),
        ("the cat sat on a mat.", -0.2),
    ]
    assert mbr.select(candidates) == 1


# 무작위로 지은 후보 묶음(두 언어, 2~8개, 겹치는 글과 같은 점수 포함)에서 서비스의 고르기가 측정 도구
# eval.mbr.select(sacrebleu 효용)와 같은 위치를 고른다. 같은 시드라 늘 같은 묶음이다.
def test_select_matches_the_measurement_tool_on_random_candidates():
    pytest.importorskip("sacrebleu")
    from eval import mbr as measured

    rng = random.Random(20261103)
    for _ in range(300):
        words = WORDS if rng.random() < 0.5 else KOREAN
        count = rng.randint(2, 8)
        candidates = []
        for _ in range(count):
            if candidates and rng.random() < 0.2:
                text = candidates[-1][0]  # 겹치는 글
            else:
                text = " ".join(rng.choice(words) for _ in range(rng.randint(1, 9))) + rng.choice(
                    [".", "", "!"]
                )
            score = round(rng.uniform(-1.0, 0.0), rng.choice([1, 3]))
            candidates.append((text, score))
        assert mbr.select(candidates) == measured.select(candidates)


# ---- MBR 번역기


# n-best 후보 함수만 가진 번역기 대역: 정해 둔 문장별 후보 목록을 돌려주거나 정해 둔 예외를 던지고, 받은
# 인수를 기록한다.
class FakeInner:
    model_name = "ctranslate2/fake"

    def __init__(self, lists, error=None):
        self.lists, self.error, self.calls = lists, error, []

    def candidates(self, text, source, target, beam_size, n):
        self.calls.append((text, source, target, beam_size, n))
        if self.error:
            raise self.error
        return self.lists


# 문장마다 따로 고르고(두 번째 문장은 후보가 하나라 표를 부르지 않음) 공백 하나로 잇는다. 빔 크기와 가설 수를
# 후보 함수에 넘기고, 이름에 빔 8·8개·MBR을 적는다(측정 보고서의 팔 이름과 같은 꼴).
def test_mbr_translator_picks_per_sentence_and_joins_with_one_space():
    inner = FakeInner([[("하나.", -0.1), ("일.", -0.2)], [("둘.", -0.3)]])
    matrix = table_matrix([[0, 1], [9, 0]])
    translator = mbr.MbrTranslator(inner, beam=8, n=8, matrix=matrix)
    assert translator.translate("One. Two.", "en", "ko") == "일. 둘."
    assert inner.calls == [("One. Two.", "en", "ko", 8, 8)]
    assert matrix.calls == [["하나.", "일."]]
    assert translator.model_name == "ctranslate2/fake/b8-mbr8"


# 후보 함수의 ModelError(엔진 오류·빈 번역·다른 방향)는 그대로 올라가고, 고르기 안의 다른 예외는 번역
# 실패(ModelError)로 바뀐다(API 503). 둘 다 번역기에 남는 상태가 없다.
def test_mbr_translator_error_boundary():
    engine_error = ModelError("the translation model failed: RuntimeError")
    with pytest.raises(ModelError) as raised:
        mbr.MbrTranslator(FakeInner([], error=engine_error)).translate("x", "ko", "en")
    assert raised.value is engine_error

    def broken(texts):
        raise RuntimeError("boom")

    with pytest.raises(ModelError, match="MBR selection failed") as raised:
        mbr.MbrTranslator(FakeInner([[("a", 0.0), ("b", -1.0)]]), matrix=broken).translate("x", "ko", "en")
    assert isinstance(raised.value.__cause__, RuntimeError)


# 가설 수는 1 이상 빔 크기 이하만 받는다(설정 실수는 만들 때 멈춤).
@pytest.mark.parametrize(("beam", "n"), [(4, 8), (8, 0)])
def test_mbr_translator_needs_between_one_and_beam_hypotheses(beam, n):
    with pytest.raises(ValueError, match="hypotheses"):
        mbr.MbrTranslator(FakeInner([]), beam=beam, n=n)


# SentencePiece 대역: 단어 하나가 조각 하나.
class Words:
    def encode(self, text, out_type):
        return text.split()

    def decode(self, pieces):
        return " ".join(pieces)


# CTranslate2 n-best 대역: 문장의 첫 토큰으로 정해 둔 가설 목록을 골라 n개까지 돌려주고, 1-best 경로는 부르면
# 실패해 MBR 번역기가 그 경로를 쓰지 않음을 확인한다.
class Engine:
    def __init__(self, nbest):
        self.nbest = nbest

    def generate(self, tokens, target_prefix=None):
        raise AssertionError("the MBR translator must use the n-best path")

    def generate_nbest(self, tokens, beam_size, n):
        return [(text.split(), score) for text, score in self.nbest[tokens[0]]][:n]


# 실제 MarianTranslator(한→영, 입력 통째로 = 서비스 구성)에 얹어: 입력 전체의 n-best를 받아 실제 chrF로
# 고르고, 이름은 'ctranslate2/<모델 폴더>/b8-mbr8'이다. 다른 방향은 감싼 번역기가 ModelError로 거절한다.
def test_mbr_translator_on_the_real_marian_translator(monkeypatch):
    engine = Engine(
        {
            "그": [
                ("a dog barked loudly at night.", -0.1),
                ("the cat sat on the mat.", -0.3),
                ("the cat sat on the mat.", -0.4),
                ("the cat sat on a mat.", -0.2),
            ]
        }
    )
    monkeypatch.setattr(translation, "_Ct2Model", lambda *args: engine)
    monkeypatch.setattr(translation, "load_sentencepiece", lambda path: Words())
    inner = translation.MarianTranslator(Path("opus-mt-tc-big-ko-en"), "ko", "en", beam_size=8)
    translator = mbr.MbrTranslator(inner, beam=8, n=8)
    assert translator.translate("그 고양이가 앉았다.", "ko", "en") == "the cat sat on the mat."
    assert translator.model_name == "ctranslate2/opus-mt-tc-big-ko-en/b8-mbr8"
    with pytest.raises(ModelError, match="only translates ko->en"):
        translator.translate("The cat sat.", "en", "ko")

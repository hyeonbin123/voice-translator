"""MBR 디코딩 시험(docs/experiments.md 20절)의 후보 고르기·번역기·판정을 지어낸 후보와 보고서로 검증한다.

sacrebleu(eval 그룹)가 필요하다. CI의 일반 테스트 단계는 이 모듈을 건너뛰고, sacrebleu·pyarrow를 더한 별도
단계가 이 모듈을 돌린다.
"""

# [파일 역할] eval/mbr.py를 모델 없이 검증한다. 효용(문장 chrF)이 sacrebleu와 같은지, MBR이 다른 후보들과 견준
# 평균 효용이 가장 큰 후보를 고르는지(자기 자신은 빼고, 동점이면 모델 점수, 그래도 같으면 앞 후보), 번역기가
# 문장마다 고르고 서비스처럼 공백 하나로 잇는지, mt_eval의 팔이 등록한 디코딩으로 만들어지는지, 판정 함수가
# 규칙의 경계값(두 지표 구간 하한, 출력 점검, p50 증가 +50ms·p95 0.8초, B8 대비 조건)을 지키는지 본다.
# [왜 중요한가] 측정 전에 적은 규칙을 코드로 고정한 도구라, 같은 입력이면 늘 같은 후보와 같은 판정이 나와야
# 하고 문서의 규칙과 한 글자도 달라서는 안 된다. 실제 번역 모델은 돌리지 않는다(가짜 CT2 결과와 가짜 효용 표).

import argparse
import json
from pathlib import Path

import pytest

# chrF 계산에 sacrebleu가 필요해 없으면 파일 전체를 건너뛴다. CI는 sacrebleu를 얹은 별도 단계에서 이 파일을
# 돌린다(.github/workflows/ci.yml). 그래서 아래 import들은 이 줄 뒤에 E402 경고를 끄는 표시와 함께 있다.
pytest.importorskip("sacrebleu")

from app.services import translation  # noqa: E402
from app.services.interfaces import ModelError  # noqa: E402
from eval import comet_score, mbr  # noqa: E402
from eval.mt_checks import check_items  # noqa: E402
from eval.mt_decide import comet_key  # noqa: E402

# 테스트 속도를 위해 붓스트랩 반복을 실제(10,000번)보다 줄인다. WORDS는 서로 다른 시험 문장을 만들 단어
# 목록이다(test_mt_decide.py와 같은 방식).
ROUNDS = 300
WORDS = "the river runs past the old mill and the children wave at the boats every summer morning".split()


# ---- 효용과 MBR 고르기


# 정해 둔 효용 표를 돌려주는 대역. 받은 후보 글 목록을 calls에 남긴다(표를 부르지 않아야 하는 경우를 확인).
def table_matrix(table):
    calls = []

    def matrix(texts):
        calls.append(list(texts))
        return table

    matrix.calls = calls
    return matrix


# 빠른 길(후보마다 n-gram을 한 번만 뽑아 짝마다 일치 수만 셈)이 sacrebleu의 공개 함수 sentence_score(2절과
# 같은 기본 설정)와 소수점 끝자리까지 같아야 한다. 한국어·영어·문장부호·연속 공백·빈 글·같은 글을 모두 짝지어
# 본다.
def test_chrf_matrix_is_sacrebleu_sentence_chrf_with_the_default_settings():
    from sacrebleu.metrics import CHRF

    texts = [
        "비가 와서 집에 있었다.",
        "비가 와서 집에 머물렀다.",
        "It rained, so I stayed home.",
        "집에 있었다",
        "",
        "  띄어  쓰기 ",
        "(괄호) 안의 말!",
        "비가 와서 집에 있었다.",
    ]
    table = mbr.chrf_matrix(texts)
    metric = CHRF()
    for i, hypothesis in enumerate(texts):
        for j, reference in enumerate(texts):
            if i != j:
                assert table[i][j] == metric.sentence_score(hypothesis, [reference]).score


# 실제 chrF로: 엔진의 1-best(점수 −0.1)가 다른 후보들과 동떨어진 글이면, 서로 가까운 후보들 쪽을 고른다. 같은
# 글이 둘이면 효용이 같아 모델 점수가 높은 쪽(위치 1)이다. 겹치는 글도 따로 센다(가설 하나가 의사 참조 하나).
def test_select_prefers_the_consensus_over_the_engine_1_best():
    candidates = [
        ("a dog barked loudly at night.", -0.1),
        ("the cat sat on the mat.", -0.3),
        ("the cat sat on the mat.", -0.4),
        ("the cat sat on a mat.", -0.2),
    ]
    assert mbr.select(candidates) == 1


# 효용이 같으면 모델 점수가 높은 후보, 그것도 같으면 앞 후보(엔진이 준 순서).
def test_select_breaks_utility_ties_by_model_score_then_by_order():
    tie = [[0, 5, 5], [5, 0, 5], [5, 5, 0]]
    assert mbr.select([("a", -0.5), ("b", -0.1), ("c", -0.3)], table_matrix(tie)) == 1
    assert mbr.select([("a", -0.2), ("b", -0.2), ("c", -0.2)], table_matrix(tie)) == 0


# 자기 자신과의 효용(표의 대각선)은 쓰지 않는다. 대각선을 넣으면 후보 0이 뽑히는 표다.
def test_select_leaves_each_candidate_out_of_its_own_pseudo_references():
    table = [[1000, 1, 1], [50, 0, 40], [50, 40, 0]]
    assert mbr.select([("a", 0.0), ("b", -1.0), ("c", -2.0)], table_matrix(table)) == 1


# 평균은 더하는 순서와 상관없어야 한다: 후보 0과 1은 같은 값(0.1, 0.2, 0.3)을 다른 순서로 받으므로 효용이
# 같고, 모델 점수가 높은 1이 뽑힌다. 왼쪽부터 그냥 더하면 0.6000000000000001과 0.6이 되어 0이 뽑힌다.
def test_select_sees_equal_utilities_whatever_the_order_of_the_others():
    table = [[0, 0.1, 0.2, 0.3], [0.3, 0, 0.2, 0.1], [0.0, 0.0, 0, 0.0], [0.0, 0.0, 0.0, 0]]
    assert mbr.select([("a", -0.9), ("b", -0.1), ("c", 0.0), ("d", 0.0)], table_matrix(table)) == 1


# 후보가 하나면 효용을 계산하지 않고 그대로 고른다. 후보가 없으면 ValueError(후보 함수가 빈 목록을 내지 않음).
def test_select_returns_a_single_candidate_without_scoring():
    matrix = table_matrix([[0]])
    assert mbr.select([("only", -1.0)], matrix) == 0
    assert matrix.calls == []
    with pytest.raises(ValueError):
        mbr.select([], matrix)


# ---- MBR 번역기


# n-best 후보 함수만 가진 번역기 대역: 정해 둔 문장별 후보 목록을 돌려주거나 정해 둔 예외를 던지고, 받은
# 인수를 기록한다.
class FakeInner:
    model_name = "ctranslate2/fake"
    by_sentence = True

    def __init__(self, lists, error=None):
        self.lists, self.error, self.calls = lists, error, []

    def candidates(self, text, source, target, beam_size, n):
        self.calls.append((text, source, target, beam_size, n))
        if self.error:
            raise self.error
        return self.lists


# 문장마다 따로 고르고(두 번째 문장은 후보가 하나라 표를 부르지 않음) 공백 하나로 잇는다. 빔 크기와 가설 수를
# 후보 함수에 넘기고, 고른 위치를 문장마다 남기며, 이름과 디코딩 기록에 빔 8·8개·MBR을 적는다.
def test_mbr_translator_picks_per_sentence_and_joins_with_one_space():
    inner = FakeInner([[("하나.", -0.1), ("일.", -0.2)], [("둘.", -0.3)]])
    matrix = table_matrix([[0, 1], [9, 0]])
    translator = mbr.MbrTranslator(inner, beam=8, n=8, matrix=matrix)
    assert translator.translate("One. Two.", "en", "ko") == "일. 둘."
    assert inner.calls == [("One. Two.", "en", "ko", 8, 8)]
    assert matrix.calls == [["하나.", "일."]]
    assert translator.picks == [{"candidates": 2, "chosen": 1}, {"candidates": 1, "chosen": 0}]
    assert translator.model_name == "ctranslate2/fake/b8-mbr8"
    assert translator.decoding == {
        "beam_size": 8,
        "num_hypotheses": 8,
        "selection": "mbr-chrf",
        "by_sentence": True,
    }


# 후보 함수의 ModelError는 그대로 올라가고(측정 도구가 실패로 셈), 앞 번역에서 고른 위치가 남지 않는다.
def test_mbr_translator_passes_model_errors_on_and_keeps_no_stale_picks():
    inner = FakeInner([[("a", 0.0), ("b", -1.0)]])
    translator = mbr.MbrTranslator(inner, matrix=table_matrix([[0, 1], [0, 0]]))
    translator.translate("x", "en", "ko")
    assert translator.picks == [{"candidates": 2, "chosen": 0}]
    inner.error = ModelError("the translation model failed: RuntimeError")
    with pytest.raises(ModelError):
        translator.translate("x", "en", "ko")
    assert translator.picks == []


# 가설 수는 1 이상 빔 크기 이하만 받는다.
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


# CTranslate2 n-best 대역: 문장의 첫 토큰으로 정해 둔 가설 목록을 골라 n개까지 돌려준다.
class Engine:
    def __init__(self, nbest):
        self.nbest = nbest

    def generate_nbest(self, tokens, beam_size, n):
        return [(text.split(), score) for text, score in self.nbest[tokens[0]]][:n]


# 실제 MarianTranslator(문장 단위)에 얹어: 문장마다 n-best를 받아 실제 chrF로 고르고 서비스처럼 공백 하나로
# 잇는다.
def test_mbr_translator_on_the_real_marian_translator(monkeypatch):
    engine = Engine(
        {
            "It": [
                ("a dog barked loudly at night.", -0.1),
                ("the cat sat on the mat.", -0.3),
                ("the cat sat on the mat.", -0.4),
                ("the cat sat on a mat.", -0.2),
            ],
            "Bye.": [("안녕.", -0.5)],
        }
    )
    monkeypatch.setattr(translation, "_Ct2Model", lambda *args: engine)
    monkeypatch.setattr(translation, "load_sentencepiece", lambda path: Words())
    inner = translation.MarianTranslator(Path("model"), "en", "ko", beam_size=8, by_sentence=True)
    translator = mbr.MbrTranslator(inner, beam=8, n=8)
    assert translator.translate("It sat. Bye.", "en", "ko") == "the cat sat on the mat. 안녕."
    assert translator.picks == [{"candidates": 4, "chosen": 1}, {"candidates": 1, "chosen": 0}]


# ---- mt_eval의 팔


# mt_eval의 여섯 팔이 판정 함수가 기다리는 디코딩으로 만들어지고(기준 빔 4·B8 빔 8의 1-best, B8-MBR 빔 8의
# 8-best + MBR, 영→한은 셋 다 문장 단위), 한 방향의 세 팔이 같은 모델 폴더를 쓰는지 본다. 모델 파일은 읽지
# 않는다.
def test_mt_eval_arms_build_the_registered_decodings(monkeypatch):
    pytest.importorskip("pyarrow")
    from eval import mt_eval

    monkeypatch.setattr(translation, "_Ct2Model", lambda *args: object())
    monkeypatch.setattr(translation, "load_sentencepiece", lambda path: Words())
    for direction, names in mbr.ARMS.items():
        models = []
        for name in names:
            factory, directions = mt_eval.CANDIDATES[name]
            assert directions == [tuple(direction.split("-"))]
            translator = factory()
            assert translator.decoding == mbr.DECODING[name]
            is_mbr = isinstance(translator, mbr.MbrTranslator)
            assert is_mbr == name.endswith("/b8-mbr")
            models.append((translator.inner if is_mbr else translator).model_name)
        assert len(set(models)) == 1


# mt_eval이 팔의 디코딩 기록과 MBR이 문장마다 고른 위치를 보고서에 남긴다(MBR 계산은 번역 시간 안).
def test_mt_eval_records_the_decoding_and_the_mbr_choices(monkeypatch):
    pytest.importorskip("pyarrow")
    from eval import mt_eval

    inner = FakeInner([[("하나", -0.1), ("일", -0.2)]])
    inner.by_sentence = False

    def factory():
        return mbr.MbrTranslator(inner, matrix=table_matrix([[0, 1], [9, 0]]))

    monkeypatch.setattr(mt_eval, "CANDIDATES", {"fake/b8-mbr": (factory, [("en", "ko")])})
    monkeypatch.setattr(mt_eval, "gpu_memory_mb", lambda: ("GPU", 0.0))
    pairs = [{"id": 1, "ko": "일", "en": "One"}, {"id": 2, "ko": "둘", "en": "Two"}]
    result = mt_eval.evaluate("fake/b8-mbr", pairs)
    assert result["decoding"] == {
        "beam_size": 8,
        "num_hypotheses": 8,
        "selection": "mbr-chrf",
        "by_sentence": False,
    }
    items = result["directions"]["en-ko"]["items"]
    assert [item["hypothesis"] for item in items] == ["일", "일"]
    assert [item["mbr"] for item in items] == [[{"candidates": 2, "chosen": 1}]] * 2


# 보고서 머리말은 빔 8 팔이 있을 때만 그 디코딩을 밝힌다(없으면 지금까지의 머리말 그대로).
@pytest.mark.parametrize(
    ("model", "named"), [("opus-mt-tc-big-en-ko/split/b8-mbr", True), ("opus-mt-ko-en", False)]
)
def test_mt_eval_report_names_the_beam_8_arms_only_when_they_ran(monkeypatch, tmp_path, model, named):
    pytest.importorskip("pyarrow")
    from eval import mt_eval

    monkeypatch.setattr(mt_eval, "REPORTS", tmp_path)
    args = argparse.Namespace(tag="t", split="validation")
    path = mt_eval.write_report([{"model": model, "vram_mb": 0, "directions": {}}], args, "GPU", [])
    assert ("빔 8" in Path(path).read_text(encoding="utf-8")) is named


# ---- 판정


# 단어 목록에서 길이가 다른 문장 count개를 만든다(끝의 번호로 모두 다른 문장이 된다).
def sentences(count: int) -> list[str]:
    return [" ".join(WORDS[i % 7 : i % 7 + 6 + i % 4]) + f" {i}." for i in range(count)]


# 끝 단어 k개를 뺀 번역(빼는 단어가 많을수록 chrF가 낮다).
def drop_words(references, k):
    return [" ".join(reference.split()[:-k]) for reference in references]


# mt_eval 보고서의 한 방향 결과 모양. 출력 점검은 실제 check_items로 계산해 판정이 실제 점검 결과를 쓰게 한다.
# 원문에 방향을 넣어 두 방향의 COMET 키(원문·번역문·참조)가 겹치지 않게 한다.
def values(hypotheses, references, p50=0.2, p95=0.5, direction="en-ko"):
    import sacrebleu

    items = [
        {"id": i, "source": f"{direction} source {i}", "reference": r, "hypothesis": h}
        for i, (r, h) in enumerate(zip(references, hypotheses, strict=True))
    ]
    return {
        "chrf": sacrebleu.corpus_chrf(hypotheses, [references]).score,
        "latency_p50": p50,
        "latency_p95": p95,
        "checks": check_items(items, direction),
        "items": items,
    }


# 한 결과의 모든 문장에 같은 COMET 점수를 주는 점수표(키는 원문·번역문·참조로 만든 comet_key).
def comet_for(direction_values: dict, score: float) -> dict[str, float]:
    return {comet_key(i["source"], i["hypothesis"], i["reference"]): score for i in direction_values["items"]}


# 팔이 지금 설정보다 두 지표 모두 낫고, 출력 점검이 늘지 않고, p50 증가가 +50ms 안이고 p95가 0.8초 안이면
# 통과한다. 경계값(+50ms, 0.8초)은 통과이고, 부동소수점 표기 오차(0.55 − 0.5 = 0.050000000000000044)로
# 떨어지지 않는다.
@pytest.mark.parametrize(
    ("p50", "p95", "ok"), [(0.55, 0.8, True), (0.5501, 0.5, False), (0.5, 0.8005, False)]
)
def test_judge_time_limits(p50, p95, ok):
    refs = sentences(30)
    base = values(drop_words(refs, 2), refs, p50=0.5, p95=0.6)
    arm = values(refs, refs, p50=p50, p95=p95)
    comet = {**comet_for(arm, 0.9), **comet_for(base, 0.7)}
    result = mbr.judge(arm, base, comet, ROUNDS)
    assert result["rules"]["latency"] is ok
    assert result["rules"]["quality"] and result["rules"]["checks"]
    assert result["passes"] is ok


# 품질은 chrF와 COMET-22 차이의 95% 구간 하한이 둘 다 0보다 커야 한다: chrF만 나아지면 떨어진다.
def test_judge_quality_needs_both_intervals_above_zero():
    refs = sentences(40)
    base = values(drop_words(refs, 2), refs)
    arm = values(refs, refs)
    same = {**comet_for(arm, 0.7), **comet_for(base, 0.7)}
    result = mbr.judge(arm, base, same, ROUNDS)
    assert result["chrf"]["low"] > 0 and result["comet"]["low"] == 0
    assert result["rules"]["quality"] is False and result["passes"] is False


# 출력 점검의 다른 문자·설명형 수와 빈 출력 수는 각각 지금 설정보다 늘면 안 된다(같으면 통과). 기준에 다른
# 문자(한자 '東京') 하나가 있을 때, 팔도 하나면 통과, 둘이면 실패, 다른 문자는 같아도 빈 출력이 하나 생기면
# 실패.
def test_judge_output_checks_may_not_rise_above_the_baseline():
    refs = sentences(40)
    flagged = ["東京 " + refs[0]]
    base = values(flagged + drop_words(refs, 2)[1:], refs)
    outcomes = []
    for hyps in (
        flagged + refs[1:],
        flagged + ["東京 " + refs[1]] + refs[2:],
        flagged + [""] + refs[2:],
    ):
        arm = values(hyps, refs)
        comet = {**comet_for(arm, 0.9), **comet_for(base, 0.7)}
        outcomes.append(mbr.judge(arm, base, comet, ROUNDS)["rules"]["checks"])
    assert outcomes == [True, False, False]


# B8과 B8-MBR이 모두 통과하면 B8-MBR − B8도 두 지표 조건을 통과할 때만 B8-MBR, 아니면 더 단순한 B8. 하나만
# 통과하면 그 팔, 둘 다 떨어지면 기준 유지(None).
@pytest.mark.parametrize(
    ("b8", "mbr_passes", "beats_b8", "pick"),
    [
        (True, True, True, "b8-mbr"),
        (True, True, False, "b8"),
        (True, False, True, "b8"),
        (True, False, False, "b8"),
        (False, True, True, "b8-mbr"),
        (False, True, False, "b8-mbr"),
        (False, False, True, None),
        (False, False, False, None),
    ],
)
def test_choose_follows_the_registered_rule(b8, mbr_passes, beats_b8, pick):
    assert mbr.choose(b8, mbr_passes, beats_b8) == pick


# mt_eval 보고서의 팔 하나(이름, 디코딩 기록, 방향별 결과) 모양. decoding을 주지 않으면 등록한 값을 쓴다.
def run(name, direction, direction_values, decoding=None):
    return {
        "model": name,
        "model_name": "ctranslate2/fake",
        "vram_mb": 512,
        "decoding": dict(mbr.DECODING[name]) if decoding is None else decoding,
        "directions": {direction: direction_values},
    }


# 두 방향의 여섯 팔과 COMET 점수표를 만든다. 한→영은 B8(끝 단어 하나 뺌)도 B8-MBR(B8과 같은 출력)도 기준(끝 두
# 단어 뺌)보다 COMET이 낫지 않아 떨어지고, 영→한은 mbr_hyps로 B8-MBR의 출력을 정한다(기본은 참조 그대로).
def six_arms(mbr_hyps=None, mbr_p50=0.21):
    refs = sentences(40)
    out, comet = [], {}
    for direction in ("ko-en", "en-ko"):
        base = values(drop_words(refs, 2), refs, p50=0.2, direction=direction)
        b8 = values(drop_words(refs, 1), refs, p50=0.21, direction=direction)
        if direction == "ko-en":
            mbr_values = values(drop_words(refs, 1), refs, p50=0.23, direction=direction)
            scores = (0.7, 0.7, 0.7)
        else:
            mbr_values = values(mbr_hyps or refs, refs, p50=mbr_p50, direction=direction)
            scores = (0.7, 0.75, 0.8)
        for item in mbr_values["items"]:
            item["mbr"] = [{"candidates": 8, "chosen": 1 if item["id"] % 4 == 0 else 0}]
        base_name, b8_name, mbr_name = mbr.ARMS[direction]
        for name, direction_values, score in zip(
            (base_name, b8_name, mbr_name), (base, b8, mbr_values), scores, strict=True
        ):
            out.append(run(name, direction, direction_values))
            comet.update(comet_for(direction_values, score))
    return out, comet


# 영→한: B8·B8-MBR 모두 통과하고 B8-MBR − B8도 두 지표 모두 유의하게 나아 B8-MBR. 한→영: 둘 다 COMET이 기준과
# 같아 떨어져 기준 유지. 참고 칸에 MBR이 n-best 첫 후보가 아닌 것을 고른 문장 수와 B8과 출력이 같은 입력 수가
# 남는다.
def test_decide_picks_mbr_when_it_also_beats_b8():
    results, comet = six_arms()
    out = mbr.decide(results, comet, ROUNDS)["directions"]
    en_ko, ko_en = out["en-ko"], out["ko-en"]
    assert en_ko["pick"] == "opus-mt-tc-big-en-ko/split/b8-mbr"
    assert en_ko["mbr_vs_b8"]["both_above_zero"] and en_ko["mbr_vs_b8"]["used"]
    assert all(arm["passes"] for arm in en_ko["arms"].values())
    assert en_ko["mbr_choices"] == {"inputs": 40, "sentences": 40, "not_first": 10, "same_as_b8": 0}
    assert ko_en["pick"] is None
    assert not any(arm["passes"] for arm in ko_en["arms"].values())
    assert ko_en["mbr_vs_b8"]["used"] is False
    assert ko_en["mbr_choices"]["same_as_b8"] == 40


# 둘 다 통과해도 B8-MBR이 B8과 같은 출력이면(차이 0, 하한 > 0이 아님) 더 단순한 B8을 고른다.
def test_decide_keeps_the_simpler_b8_when_mbr_does_not_beat_it():
    refs = sentences(40)
    results, comet = six_arms(mbr_hyps=drop_words(refs, 1))
    # 같은 글은 COMET 키가 같아 점수가 하나다: B8과 B8-MBR 모두 0.8로 기준(0.7)보다 낫다.
    out = mbr.decide(results, comet, ROUNDS)["directions"]["en-ko"]
    assert all(arm["passes"] for arm in out["arms"].values())
    assert out["mbr_vs_b8"]["both_above_zero"] is False
    assert out["pick"] == "opus-mt-tc-big-en-ko/split/b8"


# B8-MBR만 통과하면(여기서는 B8이 기준과 같은 출력이라 품질에서 떨어짐) B8-MBR을 고른다. B8-MBR이 시간에서만
# 떨어지면(p50 +60ms) 통과한 B8을 고른다.
def test_decide_with_one_passing_arm_or_none():
    results, comet = six_arms()
    for result in results:
        if result["model"] == "opus-mt-tc-big-en-ko/split/b8":
            base = next(r for r in results if r["model"] == "opus-mt-tc-big-en-ko/split")
            result["directions"]["en-ko"] = json.loads(json.dumps(base["directions"]["en-ko"]))
    assert mbr.decide(results, comet, ROUNDS)["directions"]["en-ko"]["pick"] == (
        "opus-mt-tc-big-en-ko/split/b8-mbr"
    )
    results, comet = six_arms(mbr_p50=0.26)
    slow = mbr.decide(results, comet, ROUNDS)["directions"]["en-ko"]
    assert slow["arms"]["opus-mt-tc-big-en-ko/split/b8-mbr"]["rules"]["latency"] is False
    assert slow["pick"] == "opus-mt-tc-big-en-ko/split/b8"


# 등록한 측정이 아니면 판정하지 않는다: 팔이 빠졌거나, 같은 팔이 두 번 있거나, 팔의 디코딩 기록이 등록한 것과
# 다르거나(예: 영→한 B8이 문장 단위가 아님), 두 팔의 문장(ID·원문·참조)이 다르면 SystemExit.
def test_decide_refuses_anything_but_the_registered_six_arms():
    results, comet = six_arms()
    with pytest.raises(SystemExit, match="no opus-mt-tc-big-ko-en/b8 "):
        mbr.decide([r for r in results if r["model"] != "opus-mt-tc-big-ko-en/b8"], comet, ROUNDS)
    with pytest.raises(SystemExit, match="once"):
        mbr.decide(results + results[:1], comet, ROUNDS)
    wrong = json.loads(json.dumps(results))
    for result in wrong:
        if result["model"] == "opus-mt-tc-big-en-ko/split/b8":
            result["decoding"]["by_sentence"] = False
    with pytest.raises(SystemExit, match="decoded with"):
        mbr.decide(wrong, comet, ROUNDS)
    moved = json.loads(json.dumps(results))
    moved[-1]["directions"]["en-ko"]["items"][0]["reference"] = "another reference"
    with pytest.raises(SystemExit, match="same sentences"):
        mbr.decide(moved, comet, ROUNDS)


# COMET-22 점수 파일의 메타데이터(eval/comet_score.py가 씀). 13절에 고정한 모델·리비전·체크포인트 해시다.
def comet_meta(**changes):
    meta = {
        "model": comet_score.MODEL,
        "revision": comet_score.REVISION,
        "checkpoint_sha256": comet_score.CHECKPOINT_SHA256,
        "encoder": comet_score.ENCODER,
        "encoder_revision": comet_score.ENCODER_REVISION,
        "device": "cuda",
        "precision": "fp16",
    }
    return {**meta, **changes}


# 명령줄: 보고서와 COMET 점수 파일을 읽어 판정을 reports/mbr_decide_<태그>_<시각>.json·.md로 쓴다. 고정한
# 모델이 아닌 COMET 점수(다른 리비전)로는 판정하지 않는다.
def test_decide_command_writes_the_decision(monkeypatch, tmp_path):
    results, comet = six_arms()
    report = tmp_path / "mt_t_dev.json"
    report.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    scores = tmp_path / "comet.json"
    scores.write_text(json.dumps({"meta": comet_meta(), "scores": comet}), encoding="utf-8")
    monkeypatch.setattr(mbr, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(mbr, "ROUNDS", ROUNDS)
    mbr.main(["decide", "--report", str(report), "--comet", str(scores), "--tag", "test"])
    written = json.loads(
        next((tmp_path / "reports").glob("mbr_decide_test_*.json")).read_text(encoding="utf-8")
    )
    assert written["directions"]["en-ko"]["pick"] == "opus-mt-tc-big-en-ko/split/b8-mbr"
    assert written["rounds"] == ROUNDS and written["seed"] == 12345
    assert written["comet_meta"]["precision"] == "fp16"
    markdown = next((tmp_path / "reports").glob("mbr_decide_test_*.md")).read_text(encoding="utf-8")
    assert "opus-mt-tc-big-en-ko/split/b8-mbr" in markdown
    scores.write_text(json.dumps({"meta": comet_meta(revision="main"), "scores": comet}), encoding="utf-8")
    with pytest.raises(SystemExit, match="COMET"):
        mbr.main(["decide", "--report", str(report), "--comet", str(scores), "--tag", "test"])

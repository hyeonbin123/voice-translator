"""The re-analysis reads the stored reports right (no intervals are computed on them here).

It scores with sacrebleu and jiwer from the eval group, which CI's plain test step does not install: a step of
its own adds the two and runs this module (T88).
"""

# [파일 역할] 기록된 번역·음성 인식 결정의 재분석 도구 eval/reanalysis.py(docs/experiments.md 13절)를
# 검증한다. 저장된 보고서를 제대로 읽는지(관문 check), 비교 목록과 문턱이 측정 전에 적은 대로인지, COMET-22
# 채점 목록·정밀도 점검·보고서 쓰기가 맞는지 확인한다. 구간 계산 자체는 test_significance*.py가 맡는다.
# [왜 중요한가] 재분석은 모델을 다시 돌리지 않고 eval/reports/의 문장별 결과만으로 계산한다. 그래서 '저장된
# 문장으로 기록된 점수를 정확히 다시 만들 수 있다'와 '두 쪽이 같은 문장끼리 짝지어졌다'가 모든 결론의 전제다.
# [실행 환경] sacrebleu·jiwer는 eval 그룹에만 있어 없으면 파일 전체를 건너뛰고, CI는 둘을 얹은 별도 단계에서
# 이 파일을 돌린다(.github/workflows/ci.yml).

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest

# importorskip 뒤에 import해야 하므로 아래 import들은 E402 경고를 끄는 표시를 달고 이 자리에 있다.
# reanalysis에서는 비교 목록 상수(MT_COMPARISONS·STT_COMPARISONS), 비교·데이터 클래스, 단계별 함수(check,
# comet_*, analyse_*, markdown)와 보고서 읽기 도우미(mt·typo·stored)를 가져온다. rate_difference는 문장 단위
# 구간을 직접 계산해 견줄 기준이다.
jiwer = pytest.importorskip("jiwer")
pytest.importorskip("sacrebleu")
from sacrebleu.metrics import BLEU  # noqa: E402

from eval import reanalysis  # noqa: E402
from eval.reanalysis import (  # noqa: E402
    MT_COMPARISONS,
    STT_COMPARISONS,
    MtComparison,
    Segments,
    SttComparison,
    Utterances,
    analyse_mt,
    analyse_stt,
    check,
    comet_agreement,
    comet_key,
    comet_segments,
    markdown,
    mt,
    stored,
    typo,
    utterance_edits,
)
from eval.significance import rate_difference  # noqa: E402


# 13절의 관문: 저장된 문장별 결과로 다시 계산한 chrF·CER·WER가 기록된 값과 1e-9 안에서 모두 같고, 각 비교의 두
# 쪽이 같은 문장을 짝짓는지 실제 저장 보고서로 확인한다. 기록된 점수의 개수(66개)도 고정해, 보고서나 비교
# 목록이 바뀌면 바로 드러나게 한다.
def test_stored_items_reproduce_every_recorded_score():
    # The gate in docs/experiments.md 13: chrF, CER and WER recomputed from the items equal the recorded
    # values (T32 chrF with the references from the T3 reports), and each comparison pairs the same sentences.
    rows = check()
    assert [row["name"] for row in rows if not row["ok"]] == []
    # T3 7 + 2, T17 4 + 1, T32 18 + 6 + 6, T65 4, T2 6 + 2, T14 8 + 2
    assert len(rows) == 66


# 오타 평가(6절) 보고서는 오타 섞인 입력만 저장하고 깨끗한 원문·참조는 저장하지 않는다. 그래서 같은 분할의
# 번역 평가(2절) 보고서에서 같은 FLEURS 문장 id로 가져오는지 확인한다: 문장 id·원문·참조는 같고(129문장),
# 번역문만 다르다.
def test_typo_sentences_take_source_and_reference_from_t3():
    noisy = typo(reanalysis.T32_DEV, "en-ko", "C", "heavy")(stored)
    clean = mt(reanalysis.T3_DEV, "opus-mt-tc-big-en-ko", "en-ko")(stored)
    assert noisy.ids == clean.ids and len(noisy.ids) == 129
    assert noisy.sources == clean.sources and noisy.references == clean.references
    assert noisy.hypotheses != clean.hypotheses


# 비교 27개의 키가 겹치지 않고, 대표 비교의 문턱이 13절에 측정 전에 적은 값과 같은지 확인한다. 예: 번역 한→영
# NLLB 비교는 −1(1점 이내면 빠른 쪽), 영→한 문장 단위는 −0.3, 오타 교정 C는 깨끗한 입력 −0.5·오타 평균 +1.0,
# 음성 인식 turbo 대 large-v3는 1%p, 빈 결과 줄이기는 0.2%p. 전체 흐름 품질(9절)은 확인 측정이라 판단 규칙과
# 문턱이 없다(None).
def test_comparison_keys_are_unique_and_thresholds_registered():
    keys = [c.key for c in MT_COMPARISONS] + [c.key for c in STT_COMPARISONS]
    assert len(keys) == len(set(keys)) == 27
    thresholds = {c.key: c.threshold for c in MT_COMPARISONS + STT_COMPARISONS}
    assert thresholds["t3-ko-en-nllb"] == -1.0
    assert thresholds["t17-en-ko"] == -0.3
    assert thresholds["t32-en-ko-C-clean"] == -0.5 and thresholds["t32-en-ko-C-typo"] == 1.0
    assert thresholds["t2-turbo-large"] == 1.0 and thresholds["t14-b-a"] == 0.2
    assert thresholds["t65-ko-en"] is None


# 발화 하나의 편집 수 ÷ 정답 길이가 jiwer의 cer(한국어)·wer(영어)와 같은지 확인한다. 말뭉치 단위 오류율 구간은
# 발화별 편집 수와 길이를 더해 계산하므로, 발화 단위 셈이 jiwer와 같아야 기록된 오류율이 재현된다. 숫자 표기
# 차이, 단어 빠짐, 빈 인식 결과를 사례로 둔다.
@pytest.mark.parametrize(
    ("language", "reference", "hypothesis"),
    [
        ("ko", "이백만년동안", "200만년동안"),
        ("en", "the cat sat on the mat", "the cat sat on mat"),
        ("ko", "말", ""),
    ],
)
def test_utterance_edits_match_jiwer_rates(language, reference, hypothesis):
    edits, length = utterance_edits(language, reference, hypothesis)
    rate = jiwer.cer if language == "ko" else jiwer.wer
    assert edits / length == pytest.approx(rate(reference, hypothesis))


# COMET-22로 채점할 (원문, 번역문, 참조) 목록에 중복이 없고, 모든 번역 비교의 두 쪽 문장이 빠짐없이 들어
# 있는지 확인한다. 하나라도 빠지면 run이 점수 없는 문장 때문에 멈춘다.
def test_comet_segments_are_unique_and_cover_every_side():
    rows = comet_segments()
    keys = [row["key"] for row in rows]
    assert len(keys) == len(set(keys))
    known = set(keys)
    for comparison in MT_COMPARISONS:
        for load in comparison.a + comparison.b:
            segments = load(stored)
            triples = zip(segments.sources, segments.hypotheses, segments.references, strict=True)
            assert all(comet_key(*triple) in known for triple in triples)


# 정밀도 점검용 부분집합은 번역 평가(2절) validation의 tc-big 번역 두 방향(129 × 2 = 258문장)이다. GPU fp16
# 점수와 CPU fp32 점수를 이 부분집합에서 견준다.
def test_precision_subset_is_t3_tc_big_in_both_directions():
    rows = comet_segments(subset="precision")
    assert len(rows) == 258
    ko_en = mt(reanalysis.T3_DEV, "opus-mt-tc-big-ko-en", "ko-en")(stored)
    assert {row["mt"] for row in rows} >= set(ko_en.hypotheses)


# fp16과 fp32 점수의 일치 기준(문장별 최대 차이 0.01, 시스템 평균 차이 0.001)을 적용하는지 확인한다: 모두
# 0.0005 차이는 통과, 한 문장만 0.02 차이는 실패, 모두 0.002 차이는 문장 기준은 통과해도 평균 기준에서 실패.
# 두 점수 파일의 문장 집합이 다르면 ValueError다.
def test_comet_agreement_applies_the_registered_limits():
    keys = [row["key"] for row in comet_segments(subset="precision")]
    first = {key: 0.8 for key in keys}
    assert comet_agreement(first, {key: 0.8 + 0.0005 for key in keys})["passes"]
    shifted = dict(first)
    shifted[keys[0]] = 0.8 + 0.02  # one segment too far
    assert not comet_agreement(first, shifted)["passes"]
    assert not comet_agreement(first, {key: 0.8 + 0.002 for key in keys})["passes"]  # system means too far
    with pytest.raises(ValueError):
        comet_agreement(first, {key: 0.8 for key in keys[1:]})


# 회귀 테스트(커밋 18adc0a): comet-agreement 명령은 결과를 json.dumps로 출력하는데, passes가 numpy bool이면
# 직렬화에서 실패했다. 결과가 JSON으로 왕복되고 passes가 파이썬 bool인지 확인한다.
def test_comet_agreement_prints_as_json():
    # The comet-agreement command prints the result with json.dumps; a numpy bool there crashed it.
    keys = [row["key"] for row in comet_segments(subset="precision")]
    first = {key: 0.8 for key in keys}
    for second, expected in [({key: 0.8005 for key in keys}, True), ({key: 0.802 for key in keys}, False)]:
        assert json.loads(json.dumps(comet_agreement(first, second)))["passes"] is expected


# 아래 단위 테스트에 쓰는 작은 참조 4문장.
REFS = ("The cat sat on the mat.", "It is raining today.", "We meet at noon.", "Prices rose last year.")


# 저장 보고서 대신 주어진 번역문으로 Segments를 돌려주는 가짜 로더(reports 인자는 쓰지 않음).
def _segments(hypotheses):
    return lambda reports: Segments((1, 2, 3, 4), ("s1", "s2", "s3", "s4"), REFS, tuple(hypotheses))


# 번역 비교 한 행: a가 참조에 더 가까우니 chrF 차이가 양수이고 문턱 −1보다 구간 전체가 위라 'robust', p값이
# 계산되며, COMET 차이는 (0.9 − 0.7) × 100 = 20점이고 chrF와 부호가 같은지 확인한다. markdown 보고서 줄에
# 단계·분할·방향·문장 수와 한국어 분류 이름('견고')이 나오는지도 본다.
def test_analyse_mt_reports_chrf_bleu_and_comet_on_the_same_resamples():
    a = _segments(
        ["The cat sat on the mat.", "It rains today.", "We meet at noon.", "Prices rose last year."]
    )
    b = _segments(["A cat sits on a mat.", "Raining.", "We meet.", "Prices went up."])
    comparison = MtComparison("x", "T3", "validation", "ko-en", "a − b", (a,), (b,), -1.0, "rule")
    comet = {}
    for load, value in [(a, 0.9), (b, 0.7)]:
        segments = load(None)
        for triple in zip(segments.sources, segments.hypotheses, segments.references, strict=True):
            comet[comet_key(*triple)] = value
    row = analyse_mt(comparison, None, comet, rounds=200, trials=200)
    assert row["chrf"]["point"] > 0 and row["chrf"]["classification"] == "robust"
    assert 0 < row["chrf"]["p_bootstrap"] <= 1
    assert row["comet"]["point"] == pytest.approx(20.0)  # ×100
    assert row["comet"]["same_sign_as_chrf"]
    text = markdown(
        {
            "created": "now",
            "rounds": 200,
            "trials": 200,
            "seed": 1,
            "sacrebleu": "2.6.0",
            "signatures": {"chrf": "c", "bleu": "b"},
            "comet": None,
            "mt": [row],
            "stt": [],
        }
    )
    assert "| T3 | validation | ko-en | a − b | 4 |" in text and "견고" in text


# 두 쪽의 문장 id가 다르면(4번 대신 5번) 짝지을 수 없으므로 ValueError로 멈춘다. 다른 문장끼리 비교한 구간을
# 막는다.
def test_analyse_mt_rejects_sides_on_different_sentences():
    a = _segments(["x"] * 4)
    other = lambda reports: Segments((1, 2, 3, 5), ("s1", "s2", "s3", "s4"), REFS, ("y",) * 4)  # noqa: E731
    comparison = MtComparison("x", "T3", "validation", "ko-en", "a − b", (a,), (other,), None, "rule")
    with pytest.raises(ValueError):
        analyse_mt(comparison, None, None, rounds=10, trials=10)


# 음성 인식 비교는 두 구간을 낸다: 녹음 단위(발화 4개를 따로 재표집)와 문장 단위(같은 문장의 녹음을 묶어
# 재표집, by_group). FLEURS는 한 문장을 여러 사람이 읽으므로 판정 분류는 문장 단위 구간을 쓴다.
# 점 추정은 40글자 중 편집 2개 차이 = 5%p이고, 문장 3개·녹음 4개를 세며, 두 구간이 rate_difference를 직접 부른
# 결과와 같은지 확인한다.
def test_analyse_stt_groups_recordings_by_sentence():
    ids = (1, 1, 2, 3)  # sentence 1 read twice
    lengths = np.array([10, 10, 10, 10])
    worse = lambda reports: Utterances(ids, ("r",) * 4, np.array([2, 2, 1, 1]), lengths)  # noqa: E731
    better = lambda reports: Utterances(ids, ("r",) * 4, np.array([1, 1, 1, 1]), lengths)  # noqa: E731
    comparison = SttComparison("y", "T2", "validation", "a − b", {"ko": worse}, {"ko": better}, 1.0, "rule")
    row = analyse_stt(comparison, None, rounds=300)
    assert row["utterance"]["point"] == pytest.approx(5.0)  # 2 more edits in 40 characters, in %p
    assert row["sentences"] == {"ko": 3} and row["languages"] == {"ko": 4}
    assert row["classification"] == row["sentence"]["classification"]
    # The sentence interval resamples the three sentences (sentence 1 with both its recordings) ...
    grouped = rate_difference(
        [(np.array([400, 100, 100]), np.array([200, 100, 100]), np.array([20, 10, 10]))], 300
    )
    assert (row["sentence"]["low"], row["sentence"]["high"]) == pytest.approx((grouped.low, grouped.high))
    # ... and the utterance interval the four recordings.
    single = rate_difference([(np.array([200, 200, 100, 100]), np.array([100] * 4), lengths)], 300)
    assert (row["utterance"]["low"], row["utterance"]["high"]) == pytest.approx((single.low, single.high))


# run 전체 흐름: 비교 목록을 작은 가짜 비교 하나씩으로 바꾸고, 관문(check)은 통과로, 보고서 폴더는 tmp_path로
# 바꾼다. 보고서 JSON에 두 행과 sacrebleu 서명(참조 1개)이 적히고, COMET 점수는 실제로 쓴 8문장만 따로
# 저장되며(쓰지 않은 'unused'는 빠짐), markdown에 음성 인식 행이 '녹음 4 (문장 3)' 형식으로 나오는지 확인한다.
# BLEU를 13a 토크나이저로 바꾼 것은 flores200 토크나이저가 처음 쓸 때 내려받기를 하기 때문이다(테스트 주석).
def test_run_writes_the_report_and_the_comet_scores_it_used(tmp_path, monkeypatch):
    a = _segments(
        ["The cat sat on the mat.", "It rains today.", "We meet at noon.", "Prices rose last year."]
    )
    b = _segments(["A cat sits on a mat.", "Raining.", "We meet.", "Prices went up."])
    ids, lengths = (1, 1, 2, 3), np.array([10, 10, 10, 10])
    worse = lambda reports: Utterances(ids, ("r",) * 4, np.array([2, 2, 1, 1]), lengths)  # noqa: E731
    better = lambda reports: Utterances(ids, ("r",) * 4, np.array([1, 1, 1, 1]), lengths)  # noqa: E731
    monkeypatch.setattr(
        reanalysis,
        "MT_COMPARISONS",
        (MtComparison("x", "T3", "validation", "ko-en", "a − b", (a,), (b,), -1.0, "rule"),),
    )
    monkeypatch.setattr(
        reanalysis,
        "STT_COMPARISONS",
        (SttComparison("y", "T2", "validation", "a − b", {"ko": worse}, {"ko": better}, 1.0, "rule"),),
    )
    monkeypatch.setattr(reanalysis, "check", lambda reports: [])
    monkeypatch.setattr(reanalysis, "bleu", BLEU)  # 13a here: flores200 would download its tokenizer
    monkeypatch.setattr(reanalysis, "REPORTS", tmp_path)
    scores = {row["key"]: 0.5 for row in comet_segments(None)}
    scores["unused"] = 0.1
    comet_file = tmp_path / "scores.json"
    comet_file.write_text(
        json.dumps(
            {
                "meta": {
                    "model": "m",
                    "revision": "r" * 40,
                    "device": "cuda",
                    "precision": "fp16",
                    "segments": 9,
                },
                "scores": scores,
            }
        )
    )
    args = argparse.Namespace(comet=str(comet_file), tag="unit", rounds=100, trials=100)
    base = reanalysis.run(args, reports=None)
    result = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
    assert [row["key"] for row in result["mt"]] == ["x"] and [row["key"] for row in result["stt"]] == ["y"]
    assert "nrefs:1" in result["signatures"]["chrf"] and "nrefs:1" in result["signatures"]["bleu"]
    kept = json.loads(Path(f"{base}_comet22.json").read_text(encoding="utf-8"))
    assert "unused" not in kept["scores"] and len(kept["scores"]) == 8
    assert "| T2 | validation | a − b | ko 4 (3) |" in base.with_suffix(".md").read_text(encoding="utf-8")


# COMET 점수 파일에 비교 문장의 점수가 하나라도 없으면 run이 SystemExit으로 멈추는지 확인한다. 일부 문장만으로
# 평균을 내는 일을 막는다.
def test_run_refuses_missing_comet_scores(tmp_path, monkeypatch):
    a = _segments(["x"] * 4)
    comparison = MtComparison("x", "T3", "validation", "ko-en", "a − b", (a,), (a,), None, "rule")
    monkeypatch.setattr(reanalysis, "MT_COMPARISONS", (comparison,))
    monkeypatch.setattr(reanalysis, "STT_COMPARISONS", ())
    monkeypatch.setattr(reanalysis, "check", lambda reports: [])
    monkeypatch.setattr(reanalysis, "REPORTS", tmp_path)
    comet_file = tmp_path / "scores.json"
    comet_file.write_text(json.dumps({"meta": {}, "scores": {}}))
    with pytest.raises(SystemExit):
        reanalysis.run(argparse.Namespace(comet=str(comet_file), tag="u", rounds=10, trials=10), reports=None)


# 회귀 테스트(커밋 4abb008): max()는 첫 값이 아닌 NaN을 건너뛰므로, NaN 점수가 정밀도 점검을 통과할 수 있었다.
# 유한한 숫자가 아닌 점수가 있으면 ValueError로 거부하는지 확인한다.
def test_comet_agreement_rejects_scores_that_are_not_numbers():
    # max() skips a NaN that is not first, so a NaN segment could otherwise pass the precision check.
    keys = [row["key"] for row in comet_segments(subset="precision")]
    first = {key: 0.8 for key in keys}
    second = dict(first)
    second[keys[5]] = float("nan")
    with pytest.raises(ValueError):
        comet_agreement(first, second)


# run도 NaN 점수가 든 파일을 SystemExit으로 거부하는지 확인한다(같은 커밋 4abb008).
def test_run_refuses_comet_scores_that_are_not_numbers(tmp_path, monkeypatch):
    a = _segments(["x"] * 4)
    comparison = MtComparison("x", "T3", "validation", "ko-en", "a − b", (a,), (a,), None, "rule")
    monkeypatch.setattr(reanalysis, "MT_COMPARISONS", (comparison,))
    monkeypatch.setattr(reanalysis, "STT_COMPARISONS", ())
    monkeypatch.setattr(reanalysis, "check", lambda reports: [])
    monkeypatch.setattr(reanalysis, "REPORTS", tmp_path)
    scores = {row["key"]: float("nan") for row in comet_segments(None)}
    comet_file = tmp_path / "scores.json"
    comet_file.write_text(json.dumps({"meta": {}, "scores": scores}))
    with pytest.raises(SystemExit):
        reanalysis.run(argparse.Namespace(comet=str(comet_file), tag="u", rounds=10, trials=10), reports=None)


# 16절 다시 재기용 바꿔 읽기: 6절 보고서(T32_DEV)의 한→영 항목 A·B·C만 새 보고서의 것으로 바뀌고, 영→한
# 항목과 다른 보고서는 그대로다. 새 보고서의 영→한 항목은 쓰지 않는다. 한→영 A·B·C가 하나씩 없으면
# ValueError로 멈춘다.
def test_the_t32_ko_en_entries_can_come_from_a_new_report():
    def entry(direction, candidate, source):
        return {"direction": direction, "candidate": candidate, "from": source}

    files = {
        reanalysis.T32_DEV: [entry("ko-en", c, "old") for c in "ABC"]
        + [entry("en-ko", c, "old") for c in "ABC"],
        "new.json": [entry("ko-en", c, "new") for c in "ABC"] + [entry("en-ko", "A", "new")],
        "other.json": ["unchanged"],
        "partial.json": [entry("ko-en", c, "new") for c in "AB"],
    }
    reports = reanalysis.with_t32_ko_en("new.json", files.__getitem__)
    read = reports(reanalysis.T32_DEV)
    assert [(e["direction"], e["candidate"], e["from"]) for e in read] == [
        *(("ko-en", c, "new") for c in "ABC"),
        *(("en-ko", c, "old") for c in "ABC"),
    ]
    assert reports("other.json") == ["unchanged"]
    with pytest.raises(ValueError):
        reanalysis.with_t32_ko_en("partial.json", files.__getitem__)


# 바꿔 읽어도 관문은 그대로 돈다: 새 보고서 자리에 기록된 6절 보고서를 그대로 넣으면 66개가 모두 재현된다.
def test_the_gate_runs_on_the_replaced_t32_entries():
    def files(name):
        return stored(reanalysis.T32_DEV if name == "same.json" else name)

    rows = check(reanalysis.with_t32_ko_en("same.json", files))
    assert [row["name"] for row in rows if not row["ok"]] == [] and len(rows) == 66


# 명령줄의 --t32-ko-en이 check·comet-input·run에 바꿔 읽는 함수를 넘기는지 확인한다(여기서는 check).
def test_the_command_line_passes_the_replacement_on(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(reanalysis, "with_t32_ko_en", lambda report: ("replaced", report))
    monkeypatch.setattr(reanalysis, "check", lambda reports: seen.append(reports) or [])
    monkeypatch.setattr(sys, "argv", ["reanalysis", "check", "--t32-ko-en", "typo_new.json"])
    reanalysis.main()
    assert seen == [("replaced", "typo_new.json")]

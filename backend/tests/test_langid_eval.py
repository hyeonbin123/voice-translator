# [파일 역할] 두 사람 대화의 언어 판별 측정 도구 eval/langid_eval.py(docs/experiments.md 10절)의 요약 함수가
# 판별이 하나도 없는 조건에서도 멈추지 않고 보고서를 쓰는지 확인한다. 모델은 돌리지 않고 판별 기록은 지어낸
# 값이다.
# [왜 중요한가] 요약은 GPU로 모든 녹음을 판별한 뒤에 돈다. 한 언어의 녹음이 모두 빠지면(한 마디로 잘리지 않음)
# 예전에는 0으로 나누기에서 멈춰 그때까지의 판별 기록도 남지 않았다.

import argparse
import json

import pytest

from eval import langid_eval

THRESHOLDS = [0.6, 0.9]


# 한국어 녹음 두 개(길이마다 판별 하나씩)만 있고 영어 판별은 없는 기록. 첫 녹음은 확신 0.95로 맞혔고, 둘째는
# 확신 0.7로 영어라고 틀렸다.
def korean_only() -> list[dict]:
    return [
        {"key": key, "lang": "ko", "length": length, "guess": guess, "confidence": confidence, "seconds": 0.1}
        for key, guess, confidence in (("0-1", "ko", 0.95), ("1-2", "en", 0.7))
        for length in langid_eval.LENGTHS
    ]


# 판별이 없는 영어 조건은 녹음 0개에 비율·중앙값이 None이고, 그 조건이 있으면 규칙(모든 조건의 나머지 정확도
# 99% 이상)은 확인할 수 없어 충족하지 않는다. 전체 넘김 비율은 판별이 있는 한국어 조건만으로 계산한다
# (τ=0.9에서 6개 중 3개 넘김).
def test_a_condition_without_guesses_is_reported_not_divided_by_zero():
    conditions, rule = langid_eval.summarize(korean_only(), THRESHOLDS)
    english = [c for c in conditions if c["lang"] == "en"]
    assert [c["count"] for c in english] == [0, 0, 0]
    assert all(
        c["accuracy"] is None and c["seconds_p50"] is None and c["unsure_0.9"] is None for c in english
    )
    korean = [c for c in conditions if c["lang"] == "ko"]
    assert all(c["accuracy"] == 0.5 and c["unsure_0.9"] == 0.5 for c in korean)
    assert [r["meets"] for r in rule] == [False, False]
    assert rule[1]["unsure_overall"] == pytest.approx(0.5)


# 판별이 하나도 없으면 전체 넘김 비율도 None이다.
def test_no_guesses_at_all_give_no_overall_rate():
    _, rule = langid_eval.summarize([], THRESHOLDS)
    assert [r["unsure_overall"] for r in rule] == [None, None]


# 보고서(JSON·마크다운)가 None 칸을 '-'로 쓰고 끝까지 쓰이는지 확인한다. GPU 이름 조회와 보고서 폴더는
# 가짜로 바꾼다.
def test_the_report_is_written_with_empty_conditions(tmp_path, monkeypatch):
    monkeypatch.setattr(langid_eval, "REPORTS", tmp_path)
    monkeypatch.setattr(langid_eval, "gpu_memory_mb", lambda: ("none", 0))
    args = argparse.Namespace(tag="t", split="validation", thresholds=THRESHOLDS)
    langid_eval.write_report(korean_only(), {"ko": 0, "en": 3}, args)
    report = json.loads(next(tmp_path.glob("langid_t_*.json")).read_text(encoding="utf-8"))
    assert report["skipped"] == {"ko": 0, "en": 3}
    markdown = next(tmp_path.glob("langid_t_*.md")).read_text(encoding="utf-8")
    assert "| en | 1s | 0 | - | - | - / - | - / - |" in markdown

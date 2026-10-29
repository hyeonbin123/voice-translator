"""영→한 말투 후처리 시험(docs/experiments.md 21절)의 측정·판정 도구 eval/register_eval.py를 검증한다.

sacrebleu(eval 그룹)가 필요하다. CI의 일반 테스트 단계는 이 모듈을 건너뛰고, sacrebleu·jiwer·pyarrow를 더한
별도 단계가 이 모듈을 돌린다. 형태소 분석기는 쓰지 않는다(가짜 후처리 함수).
"""

# [파일 역할] 말투 분류(문장 끝 글자 규칙), 일관성 지표, 저장된 번역문에 후처리를 거는 run(보고서·바뀐 문장
# 목록·COMET 입력), 판정 decide(자동 관문 셋: 일관성·COMET-22 비열등·후처리 시간)를 지어낸 보고서로 확인한다.
# [무엇을 확인하나] 분류의 갈래(합쇼체 '습/ㅂ+니다'와 '아니다'를 가름, 해요체, 해라체 평서, 그 밖),
# 섞임·일관성 수, 후처리 시간은 예열 호출을 빼고 출력마다 잼, 판정 문턱의 경계값(98%·섞임 0·하한 > −0.5·p50 ≤
# 5ms), 두 출력의 문장이 다르거나 COMET 점수가 빠지거나 고정 모델이 아니면 판정하지 않음, 사람 확인 전에는
# 채택하지 않음.
# [왜 중요한가] 측정 전에 적은 규칙을 코드로 고정한 도구라 같은 입력이면 늘 같은 판정이 나와야 하고, 21절의
# 문턱과 한 글자도 달라서는 안 된다.

import json
from pathlib import Path

import pytest

# 판정의 chrF·spBLEU에 sacrebleu가 필요해 없으면 파일 전체를 건너뛴다. CI는 sacrebleu를 얹은 별도 단계에서 이
# 파일을 돌린다(.github/workflows/ci.yml). 그래서 아래 import들은 이 줄 뒤에 E402 경고를 끄는 표시와 함께
# 있다.
pytest.importorskip("sacrebleu")

from sacrebleu.metrics import BLEU  # noqa: E402

from eval import comet_score, register_eval  # noqa: E402
from eval.mt_decide import comet_key  # noqa: E402

# 테스트 속도를 위해 붓스트랩 반복을 실제(10,000번)보다 줄인다.
ROUNDS = 300
MODEL = "opus-mt-tc-big-en-ko/split"


# ---- 말투 분류와 일관성


@pytest.mark.parametrize(
    ("sentence", "kind"),
    [
        ("그는 집에 갔습니다.", "formal"),
        ("그것은 사실이 아닙니다.", "formal"),
        ("그것은 무엇입니까?", "formal"),
        ("조용히 하십시오.", "formal"),
        ("함께 갑시다.", "formal"),
        ("고마워요.", "polite"),
        ("그렇죠?", "polite"),
        ("그는 집에 갔다.", "plain"),
        ("그것은 사실이 아니다.", "plain"),  # '니다'로 끝나도 앞 음절이 습·ㅂ 받침이 아니면 해라체
        ("그는 학교에 다니다.", "plain"),
        ('그는 "나는 간다."라고 말했다."', "plain"),  # 끝의 문장부호·닫는 따옴표를 떼고 본다
        ("(그는 갔다.)", "plain"),
        ("왜 그럴까?", "other"),
        ("가자.", "other"),
        ("하지 마라.", "other"),
        ("미국의 조지 W.", "other"),
        ("그 숫자는 10.", "other"),
        ("", "other"),
    ],
)
def test_classify_end(sentence, kind):
    assert register_eval.classify_end(sentence) == kind


def test_consistency_counts_outputs_whose_every_end_is_honorific_and_mixed_outputs():
    outputs = [
        "그는 갔습니다. 그녀는 갔다.",  # 섞임
        "그는 갔습니다. 고마워요.",  # 모두 높임(합쇼체 + 해요체)
        "그는 갔다.",  # 해라체만: 섞임은 아니지만 모두 높임도 아님
        "미국의 조지 W.",  # 그 밖: 섞임 아님, 모두 높임 아님
        "왜 그럴까? 그는 갔습니다.",  # 그 밖 + 높임: 섞임 아님(해라체 평서만 섞임으로 셈)
    ]
    result = register_eval.consistency(outputs)
    assert result == {
        "outputs": 5,
        "all_honorific": 1,
        "ratio": 0.2,
        "mixed": 1,
        "ends": {"formal": 3, "polite": 1, "plain": 2, "other": 2},
    }


# ---- run: 저장된 번역문에 후처리 걸기


# eval.mt_eval 보고서 모양: 결과 목록, 결과마다 model·directions(방향 → items). 다른 모델 결과도 하나 넣는다.
def source_report(hyps: list[str]) -> list[dict]:
    items = [
        {
            "id": 1000 + i,
            "latency_s": 0.1,
            "chrf": 50.0,
            "sentences": hyp.count(".") or 1,
            "source": f"Person {i} went home.",
            "reference": f"{i}번 사람은 집에 갔습니다.",
            "hypothesis": hyp,
        }
        for i, hyp in enumerate(hyps)
    ]
    return [
        {"model": "opus-mt-tc-big-en-ko", "directions": {"en-ko": {"items": []}}},
        {"model": MODEL, "directions": {"en-ko": {"chrf": 40.0, "count": len(items), "items": items}}},
    ]


def fake_formalize(text: str) -> str:
    return text.replace("갔다.", "갔습니다.")


# 정해 둔 걸린 시간(초)을 차례로 내는 가짜 시계: 호출 하나당 시작·끝 두 번 읽는다.
def fake_clock(durations_s: list[float]):
    ticks, now = [], 0.0
    for duration in durations_s:
        ticks += [now, now + duration]
        now += duration + 1.0
    readings = iter(ticks)
    return lambda: next(readings)


def test_run_applies_the_conversion_to_every_saved_output_and_times_each():
    hyps = ["0번 사람은 집에 갔다.", "1번 사람은 집에 갔습니다.", "2번 사람은 집에 갔습니다. 그는 갔다."]
    warmed = []

    def formalize(text):
        warmed.append(text)
        return fake_formalize(text)

    report = register_eval.run(source_report(hyps), MODEL, formalize, clock=fake_clock([0.002, 0.001, 0.004]))
    asis, formal = report
    assert asis["model"] == MODEL and asis["register"] == "asis"
    assert formal["model"] == MODEL + "+formal" and formal["register"] == "formal"
    before, after = asis["directions"]["en-ko"], formal["directions"]["en-ko"]
    assert [item["hypothesis"] for item in before["items"]] == hyps
    assert [item["hypothesis"] for item in after["items"]] == [
        "0번 사람은 집에 갔습니다.",
        "1번 사람은 집에 갔습니다.",
        "2번 사람은 집에 갔습니다. 그는 갔습니다.",
    ]
    # 두 출력은 같은 문장(ID·원문·참조)을 같은 순서로 가진다
    assert [(i["id"], i["source"], i["reference"]) for i in before["items"]] == [
        (i["id"], i["source"], i["reference"]) for i in after["items"]
    ]
    assert [item["changed"] for item in after["items"]] == [True, False, True]
    assert [item["postprocess_ms"] for item in after["items"]] == [2.0, 1.0, 4.0]
    # 첫 호출은 예열(시간에 안 듦)이고, 그 뒤 출력마다 한 번씩 부른다
    assert len(warmed) == 4 and warmed[1:] == hyps
    assert after["postprocess_p50_ms"] == 2.0
    assert after["changed"] == 2 and after["count"] == 3
    # 글자 수: 바뀐 출력마다 '갔다'→'갔습니다'로 2글자씩 늘었다
    assert after["char_increase_median"] == 2.0
    assert before["consistency"]["mixed"] == 1 and after["consistency"]["mixed"] == 0
    assert after["consistency"]["all_honorific"] == 3


def test_run_needs_the_registered_output():
    with pytest.raises(SystemExit, match="no opus-mt-tc-big-en-ko/split/b8 en-ko"):
        register_eval.run(source_report(["갔다."]), MODEL + "/b8", fake_formalize)
    with pytest.raises(SystemExit, match="no outputs"):
        register_eval.run(source_report([]), MODEL, fake_formalize)


def test_the_changed_list_shows_only_changed_outputs_side_by_side():
    report = register_eval.run(
        source_report(["0번 | 사람은 갔다.", "1번 사람은 갔습니다."]), MODEL, fake_formalize
    )
    lines = register_eval.changes_lines(report)
    table = [line for line in lines if line.startswith("| 1000")]
    assert table == ["| 1000 | Person 0 went home. | 0번 \\| 사람은 갔다. | 0번 \\| 사람은 갔습니다. |  |"]
    assert not any(line.startswith("| 1001") for line in lines)


def test_run_command_writes_the_report_the_list_and_the_comet_input(monkeypatch, tmp_path):
    source = tmp_path / "mt_t17_dev.json"
    hyps = ["0번 사람은 집에 갔다.", "1번 사람은 집에 갔습니다."]
    source.write_text(json.dumps(source_report(hyps), ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(register_eval, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(register_eval, "load_formalizer", lambda: fake_formalize)
    changes = tmp_path / "work" / "changed.md"
    comet_input = tmp_path / "work" / "comet_input.jsonl"
    register_eval.main(
        [
            "run",
            "--report",
            str(source),
            "--tag",
            "test",
            "--changes",
            str(changes),
            "--comet-input",
            str(comet_input),
        ]
    )
    written = next((tmp_path / "reports").glob("register_test_*.json"))
    report = json.loads(written.read_text(encoding="utf-8"))
    assert [entry["model"] for entry in report] == [MODEL, MODEL + "+formal"]
    assert report[0]["source_report"] == str(source)
    assert next((tmp_path / "reports").glob("register_test_*.md")).exists()
    assert "0번 사람은 집에 갔습니다." in changes.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in comet_input.read_text(encoding="utf-8").splitlines()]
    # 바뀌지 않은 출력(1번)은 두 쪽이 같은 세 쌍이라 한 번만 채점한다
    assert len(rows) == 3
    assert {row["mt"] for row in rows} == {hyps[0], hyps[1], "0번 사람은 집에 갔습니다."}


# ---- 판정


@pytest.mark.parametrize(
    ("all_honorific", "mixed", "comet_low", "p50_ms", "rules"),
    [
        (98, 0, -0.49, 5.0, (True, True, True)),  # 경계값: 98%·하한 −0.49·p50 5ms는 통과
        (97, 0, 0.0, 1.0, (False, True, True)),  # 일관성 97%
        (100, 1, 0.0, 1.0, (False, True, True)),  # 섞인 출력 하나
        (100, 0, -0.5, 1.0, (True, False, True)),  # 하한이 −0.5와 같으면 실패(> −0.5여야 함)
        (100, 0, 0.0, 5.0004, (True, True, True)),  # ms 소수 셋째 자리로 반올림해 견줌
        (100, 0, 0.0, 5.001, (True, True, False)),
    ],
)
def test_judge_follows_the_registered_thresholds(all_honorific, mixed, comet_low, p50_ms, rules):
    after = {"outputs": 100, "all_honorific": all_honorific, "mixed": mixed}
    result = register_eval.judge(after, comet_low, p50_ms)
    assert (result["consistency"], result["comet"], result["latency"]) == rules


# run 결과(asis·formal 두 출력)와 그 문장들의 COMET 점수표. formal_score는 바뀐 출력의 COMET 점수다.
def scored_report(formal_score: float, size: int = 40):
    hyps = [f"{i}번 사람은 집에 갔다." if i % 2 == 0 else f"{i}번 사람은 집에 갔습니다." for i in range(size)]
    report = register_eval.run(source_report(hyps), MODEL, fake_formalize)
    comet = {}
    for entry in report:
        for item in entry["directions"]["en-ko"]["items"]:
            changed = entry["register"] == "formal" and item["changed"]
            comet[comet_key(item["source"], item["hypothesis"], item["reference"])] = (
                formal_score if changed else 0.8
            )
    return report, comet


def test_decide_passes_the_automatic_gates_but_waits_for_the_human_check(monkeypatch):
    monkeypatch.setattr(register_eval, "bleu", BLEU)  # 여기서는 13a: flores200은 토크나이저를 내려받는다
    report, comet = scored_report(formal_score=0.8)
    result = register_eval.decide(report, comet, ROUNDS)
    assert result["after"]["mixed"] == 0 and result["after"]["all_honorific"] == 40
    assert result["before"]["all_honorific"] == 20
    assert result["comet"]["point"] == pytest.approx(0.0)
    assert result["rules"] == {"consistency": True, "comet": True, "latency": True}
    assert result["automatic_passes"] is True
    assert result["verdict"] == "automatic-pass-pending-human-check"
    assert result["human_check"] is None
    assert result["chrf"]["a"] > result["chrf"]["b"]  # 참조가 합쇼체라 바꾼 쪽 chrF가 높다(적기만 함)
    lines = register_eval.decide_lines(result)
    assert any("사람 문법 확인" in line for line in lines)


def test_decide_rejects_a_conversion_that_lowers_comet(monkeypatch):
    monkeypatch.setattr(register_eval, "bleu", BLEU)
    report, comet = scored_report(formal_score=0.7)  # 바뀐 20개가 10점씩 낮음 → 평균 −5점
    result = register_eval.decide(report, comet, ROUNDS)
    assert result["comet"]["high"] < -0.5
    assert result["rules"]["comet"] is False
    assert result["automatic_passes"] is False and result["verdict"] == "rejected"


def test_decide_recomputes_consistency_from_the_outputs(monkeypatch):
    # 보고서에 적힌 일관성 수가 아니라 출력 글에서 21절 분류로 다시 센다
    monkeypatch.setattr(register_eval, "bleu", BLEU)
    report, comet = scored_report(formal_score=0.8)
    report[1]["directions"]["en-ko"]["consistency"] = {"outputs": 40, "all_honorific": 0, "mixed": 40}
    assert register_eval.decide(report, comet, ROUNDS)["rules"]["consistency"] is True


def test_decide_refuses_mismatched_or_unscored_reports(monkeypatch):
    monkeypatch.setattr(register_eval, "bleu", BLEU)
    report, comet = scored_report(formal_score=0.8)
    shuffled = json.loads(json.dumps(report))
    shuffled[1]["directions"]["en-ko"]["items"].reverse()
    with pytest.raises(SystemExit, match="same sentences"):
        register_eval.decide(shuffled, comet, ROUNDS)
    first = report[1]["directions"]["en-ko"]["items"][0]
    missing = {
        key: value
        for key, value in comet.items()
        if key != comet_key(first["source"], first["hypothesis"], first["reference"])
    }
    with pytest.raises(SystemExit, match="COMET"):
        register_eval.decide(report, missing, ROUNDS)
    with pytest.raises(SystemExit, match="one asis and one formal"):
        register_eval.decide(report[:1], comet, ROUNDS)


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


def test_decide_command_writes_the_decision(monkeypatch, tmp_path):
    monkeypatch.setattr(register_eval, "bleu", BLEU)
    report, comet = scored_report(formal_score=0.8)
    path = tmp_path / "register_t_dev.json"
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    scores = tmp_path / "comet.json"
    scores.write_text(json.dumps({"meta": comet_meta(), "scores": comet}), encoding="utf-8")
    monkeypatch.setattr(register_eval, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(register_eval, "ROUNDS", ROUNDS)
    register_eval.main(["decide", "--report", str(path), "--comet", str(scores), "--tag", "test"])
    written = json.loads(
        next((tmp_path / "reports").glob("register_decide_test_*.json")).read_text(encoding="utf-8")
    )
    assert written["verdict"] == "automatic-pass-pending-human-check"
    assert written["rounds"] == ROUNDS and written["seed"] == 12345
    assert written["comet_meta"]["precision"] == "fp16"
    assert Path(written["report"]) == path
    scores.write_text(json.dumps({"meta": comet_meta(revision="main"), "scores": comet}), encoding="utf-8")
    with pytest.raises(SystemExit, match="COMET"):
        register_eval.main(["decide", "--report", str(path), "--comet", str(scores), "--tag", "test"])

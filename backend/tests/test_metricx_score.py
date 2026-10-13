"""The parts of eval/metricx_score.py that need no model (the scoring runs in its own environment)."""

# [파일 역할] eval/metricx_score.py 중 모델 없이 시험할 수 있는 부분(참조 없는 입력 문자열 만들기, 내려받은
# 파일의 SHA-256 확인)을 검증한다.
# [배경] MetricX-24는 Hy-MT2 번역기 시험에서 'FLORES 오염 가능성'을 거부권으로만 보는 참조 없는(QE)
# 점수다(docs/experiments.md 15절). 채점 자체는 backend 환경과 분리된 별도 환경에서 돌고, 이 스크립트는 맨
# 위에서 표준 라이브러리만 import하므로 backend 테스트가 불러올 수 있다(모듈 docstring).

import pytest

from eval import metricx_score


# QE 입력이 공식 코드(metricx24/predict.py --qe)와 글자 하나까지 같은 형식("source: <원문> candidate: <번역>",
# 참조 없음)인지 확인한다. 형식이 다르면 모델이 학습 때와 다른 입력을 받아 점수가 달라진다.
def test_qe_input_is_metricx24s_reference_free_format():
    # metricx24/predict.py with --qe: "source: " + source + " candidate: " + hypothesis, no reference part.
    assert metricx_score.qe_input("안녕하세요.", "Hello.") == "source: 안녕하세요. candidate: Hello."


# 가중치(pytorch_model.bin)와 토크나이저(spiece.model)가 고정한 SHA-256과 다르면 SystemExit으로 멈추는지, 둘
# 다 맞으면 가중치 해시를 돌려주는지 차례로 확인한다.
# 가중치는 pickle 파일이라 내용 확인 없이 읽으면 위험하고, 다른 리비전이면 점수가 달라지므로 해시 고정이
# 중요하다(모듈 docstring: SHA-256 확인 뒤 weights_only=True로 읽음).
# 4.9GB 실제 파일 대신 tmp_path에 가짜 파일을 만들고, monkeypatch로 고정 해시를 그 가짜 파일의 해시로 바꿔
# 가며 '가중치만 맞음 → 토크나이저에서 멈춤 → 둘 다 맞음'을 확인한다.
def test_files_other_than_the_pinned_ones_are_refused(tmp_path, monkeypatch):
    model, tokenizer = tmp_path / "model", tmp_path / "tokenizer"
    model.mkdir()
    tokenizer.mkdir()
    (model / metricx_score.WEIGHTS).write_bytes(b"not the weights")
    (tokenizer / "spiece.model").write_bytes(b"not the tokenizer")
    with pytest.raises(SystemExit, match="pytorch_model.bin"):
        metricx_score.check_files(model, tokenizer)
    monkeypatch.setattr(metricx_score, "WEIGHTS_SHA256", metricx_score.sha256(model / metricx_score.WEIGHTS))
    with pytest.raises(SystemExit, match="spiece.model"):
        metricx_score.check_files(model, tokenizer)
    monkeypatch.setattr(metricx_score, "SPIECE_SHA256", metricx_score.sha256(tokenizer / "spiece.model"))
    assert metricx_score.check_files(model, tokenizer) == metricx_score.WEIGHTS_SHA256

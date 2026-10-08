"""The parts of eval/metricx_score.py that need no model (the scoring runs in its own environment)."""

import pytest

from eval import metricx_score


def test_qe_input_is_metricx24s_reference_free_format():
    # metricx24/predict.py with --qe: "source: " + source + " candidate: " + hypothesis, no reference part.
    assert metricx_score.qe_input("안녕하세요.", "Hello.") == "source: 안녕하세요. candidate: Hello."


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

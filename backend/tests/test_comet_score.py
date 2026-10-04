"""eval/comet_score.py checks the checkpoint it scores with, not only the one it downloaded.

The script runs in its own environment (unbabel-comet, transformers<5), so torch, yaml and comet are
replaced with small stand-ins here; the file hashing and the refusal are the real code.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from eval import comet_score

CHECKPOINT_BYTES = b"a stand-in for the COMET-22 checkpoint"


class FakeModel:
    def eval(self):
        return self

    def to(self, device):
        return self

    def half(self):
        return self


def model_dirs(tmp_path: Path, content: bytes = CHECKPOINT_BYTES) -> tuple[Path, Path]:
    model_dir = tmp_path / "model"
    (model_dir / "checkpoints").mkdir(parents=True)
    (model_dir / comet_score.CHECKPOINT).write_bytes(content)
    (model_dir / "hparams.yaml").write_text("class_identifier: regression_metric\n", encoding="utf-8")
    encoder_dir = tmp_path / "encoder"
    encoder_dir.mkdir()
    return model_dir, encoder_dir


@pytest.fixture
def stand_ins(tmp_path, monkeypatch):
    """Offline snapshots in tmp_path and stand-ins for the COMET environment's packages."""
    loaded = []

    def load_from_checkpoint(**kwargs):
        loaded.append(kwargs["checkpoint_path"])
        return FakeModel()

    models = ModuleType("comet.models")
    models.str2model = {"regression_metric": SimpleNamespace(load_from_checkpoint=load_from_checkpoint)}
    comet = ModuleType("comet")
    comet.__version__ = "2.2.7"
    comet.models = models
    torch = ModuleType("torch")
    torch.__version__ = "0+stand-in"
    torch.device = lambda name: name
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    yaml = ModuleType("yaml")
    yaml.safe_load = lambda text: {"class_identifier": "regression_metric"}
    transformers = ModuleType("transformers")
    transformers.__version__ = "4.57.6"
    for name, module in {
        "comet": comet,
        "comet.models": models,
        "torch": torch,
        "yaml": yaml,
        "transformers": transformers,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    dirs = model_dirs(tmp_path)
    monkeypatch.setattr(comet_score, "snapshots", lambda offline: dirs)
    monkeypatch.setattr(comet_score, "score", lambda model, samples, device, batch_size: [0.5] * len(samples))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")  # run() sets it; monkeypatch puts the old value back
    return SimpleNamespace(model_dir=dirs[0], loaded=loaded)


def score_args(tmp_path: Path) -> argparse.Namespace:
    source = tmp_path / "input.jsonl"
    rows = [
        {"key": f"k{index}", "src": "원문", "mt": "a translation", "ref": "a reference"} for index in range(3)
    ]
    source.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    return argparse.Namespace(
        input=str(source),
        output=str(tmp_path / "out.json"),
        device="cpu",
        precision="fp32",
        batch_size=32,
        limit=None,
    )


def test_check_checkpoint_returns_the_hash_it_computed(tmp_path, monkeypatch):
    model_dir, _ = model_dirs(tmp_path)
    expected = hashlib.sha256(CHECKPOINT_BYTES).hexdigest()
    monkeypatch.setattr(comet_score, "CHECKPOINT_SHA256", expected)

    assert comet_score.check_checkpoint(model_dir) == expected


def test_check_checkpoint_refuses_another_file(tmp_path):
    model_dir, _ = model_dirs(tmp_path)

    with pytest.raises(SystemExit, match="is not the pinned"):
        comet_score.check_checkpoint(model_dir)


def test_scoring_refuses_a_checkpoint_that_is_not_the_pinned_one(tmp_path, stand_ins):
    # The pinned constant is the real COMET-22 hash; the stand-in checkpoint is something else.
    args = score_args(tmp_path)

    with pytest.raises(SystemExit, match="is not the pinned"):
        comet_score.run(args)

    assert stand_ins.loaded == []  # refused before the model is built
    assert not Path(args.output).exists()


def test_scoring_records_the_hash_of_the_checkpoint_it_read(tmp_path, monkeypatch, stand_ins):
    expected = hashlib.sha256(CHECKPOINT_BYTES).hexdigest()
    monkeypatch.setattr(comet_score, "CHECKPOINT_SHA256", expected)
    args = score_args(tmp_path)

    comet_score.run(args)

    written = json.loads(Path(args.output).read_text(encoding="utf-8"))
    assert written["meta"]["checkpoint_sha256"] == expected
    assert stand_ins.loaded == [stand_ins.model_dir / comet_score.CHECKPOINT]
    assert written["scores"] == {"k0": 0.5, "k1": 0.5, "k2": 0.5}

"""eval/comet_score.py는 내려받은 체크포인트만이 아니라 실제로 채점에 쓰는 체크포인트를 확인한다.

이 스크립트는 별도 환경(unbabel-comet, transformers<5)에서 돌기 때문에 여기서는 torch, yaml, comet을 작은
가짜로 바꾼다. 파일 해시 계산과 거절은 진짜 코드다.
"""

# [파일 역할] eval/comet_score.py(COMET-22 문장 채점)가 '채점할 때 실제로 읽은 체크포인트 파일'의 SHA-256을
# 확인하고 기록하는지 검증한다.
# [왜 필요한가] T80 재분석 기록의 checkpoint_sha256은 처음에 채점 때 계산한 값이 아니라 코드에 고정한 상수를
# 적은 것이라 확인이 실패할 수 없었다(docs/experiments.md 13절 2026-10-05 정정, T81). 그 뒤 comet_score.py는
# 적재할 때 파일을 해시해 다르면 멈추고 계산한 값을 적도록 바뀌었고, 이 테스트가 그 동작이 다시 깨지지 않게
# 지킨다.
# [설계 이유] comet_score.py는 transformers<5가 필요해 백엔드 환경이 아닌 별도 환경에서 돈다. 그래서
# torch·yaml·comet·transformers를 sys.modules에 작은 가짜 모듈로 끼워 넣고, 파일 해시와 거절 로직만 진짜
# 코드로 실행한다.

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from eval import comet_score

# 가짜 체크포인트 내용. 고정 상수(진짜 COMET-22 해시)와 해시가 다르므로 기본 상태에서는 거절돼야 한다.
CHECKPOINT_BYTES = b"a stand-in for the COMET-22 checkpoint"


# comet 모델 객체 대신 쓰는 가짜: run이 부르는 eval·to·half만 자기 자신을 돌려준다.
class FakeModel:
    def eval(self):
        return self

    def to(self, device):
        return self

    def half(self):
        return self


# tmp_path 아래에 오프라인 스냅숏 모양의 폴더(체크포인트 파일, hparams.yaml, 빈 인코더 폴더)를 만든다.
def model_dirs(tmp_path: Path, content: bytes = CHECKPOINT_BYTES) -> tuple[Path, Path]:
    model_dir = tmp_path / "model"
    (model_dir / "checkpoints").mkdir(parents=True)
    (model_dir / comet_score.CHECKPOINT).write_bytes(content)
    (model_dir / "hparams.yaml").write_text("class_identifier: regression_metric\n", encoding="utf-8")
    encoder_dir = tmp_path / "encoder"
    encoder_dir.mkdir()
    return model_dir, encoder_dir


# COMET 환경의 패키지를 가짜로 바꾸는 fixture. load_from_checkpoint가 불렸는지(loaded 목록)로 '모델을 만들기
# 전에 거절했는지'를 확인할 수 있게 한다. snapshots는 내려받기 대신 tmp 폴더를, score는 모델 대신 모든 문장에
# 0.5를 돌려준다. monkeypatch를 써서 테스트가 끝나면 sys.modules·환경 변수가 원래대로 돌아간다.
@pytest.fixture
def stand_ins(tmp_path, monkeypatch):
    """tmp_path 안의 오프라인 스냅숏과 COMET 환경 패키지의 가짜들."""
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
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")  # run()이 바꾸는 값. monkeypatch가 끝나면 원래 값으로 돌린다
    return SimpleNamespace(model_dir=dirs[0], loaded=loaded)


# run에 넘길 명령줄 인자: 문장 3개짜리 입력 JSONL, 출력 경로, CPU fp32.
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


# 고정 해시를 가짜 파일의 해시로 바꾸면 check_checkpoint가 통과하고, 상수가 아니라 직접 계산한 해시를
# 돌려주는지.
def test_check_checkpoint_returns_the_hash_it_computed(tmp_path, monkeypatch):
    model_dir, _ = model_dirs(tmp_path)
    expected = hashlib.sha256(CHECKPOINT_BYTES).hexdigest()
    monkeypatch.setattr(comet_score, "CHECKPOINT_SHA256", expected)

    assert comet_score.check_checkpoint(model_dir) == expected


# 고정 해시와 다른 파일이면 SystemExit로 멈추는지.
def test_check_checkpoint_refuses_another_file(tmp_path):
    model_dir, _ = model_dirs(tmp_path)

    with pytest.raises(SystemExit, match="is not the pinned"):
        comet_score.check_checkpoint(model_dir)


# 채점 명령 전체(run)에서도 다른 체크포인트면 멈추고, 모델을 만들기 전에(loaded가 비어 있음) 멈추며, 출력
# 파일을 쓰지 않는지. 다른 모델로 낸 점수가 파일로 남는 것을 막는다.
def test_scoring_refuses_a_checkpoint_that_is_not_the_pinned_one(tmp_path, stand_ins):
    # 고정 상수는 진짜 COMET-22 해시이고, 가짜 체크포인트는 그와 다른 파일이다.
    args = score_args(tmp_path)

    with pytest.raises(SystemExit, match="is not the pinned"):
        comet_score.run(args)

    assert stand_ins.loaded == []  # 모델을 만들기 전에 거절했다
    assert not Path(args.output).exists()


# 해시가 맞으면 채점이 끝까지 돌고, 출력 메타데이터의 checkpoint_sha256이 읽은 파일의 해시이며, 그 파일로
# 모델을 만들었고, 점수가 입력 key마다 기록되는지.
def test_scoring_records_the_hash_of_the_checkpoint_it_read(tmp_path, monkeypatch, stand_ins):
    expected = hashlib.sha256(CHECKPOINT_BYTES).hexdigest()
    monkeypatch.setattr(comet_score, "CHECKPOINT_SHA256", expected)
    args = score_args(tmp_path)

    comet_score.run(args)

    written = json.loads(Path(args.output).read_text(encoding="utf-8"))
    assert written["meta"]["checkpoint_sha256"] == expected
    assert stand_ins.loaded == [stand_ins.model_dir / comet_score.CHECKPOINT]
    assert written["scores"] == {"k0": 0.5, "k1": 0.5, "k2": 0.5}

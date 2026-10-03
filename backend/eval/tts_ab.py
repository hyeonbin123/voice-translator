"""Blind A/B listening for the Korean speech synthesis trial (T78, docs/experiments.md 12).

`build` turns the validation audio of two candidates into a local folder: index.html with 20 numbered pairs in
a shuffled order, each pair's two clips in a random order named only by number, both trimmed and
loudness-matched the same way. The answer key is written to a separate file outside that folder. The listener
opens index.html in a browser, marks each pair, and copies the result text it makes. `unblind` reads that text
with the key and counts, per model, the pairs preferred and the clips marked unacceptable. From backend/:

    uv run python -m eval.tts_ab build --tag t78_dev --a melo --b supertonic-fast \\
        --out ../work/t78/ab --key ../work/t78/ab-key.json
    uv run python -m eval.tts_ab unblind --key ../work/t78/ab-key.json --answers ../work/t78/ab/answers.txt
"""

from __future__ import annotations

import argparse
import html
import io
import json
import random
import re
import wave
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from eval.common import REPORTS

# The 20 sentences, fixed before any synthesis: random.Random(78).sample of the 129 validation sentence IDs
# (sorted), then sorted. 8 of them contain digits.
AB_IDS = (
    1513,
    1519,
    1537,
    1552,
    1560,
    1563,
    1564,
    1576,
    1582,
    1595,
    1609,
    1612,
    1617,
    1618,
    1621,
    1629,
    1631,
    1636,
    1637,
    1650,
)
SEED = 7802  # pair order and the side of each model
TARGET_DBFS = -23.0  # RMS of the frames above the floor, after trimming
FLOOR_DBFS = -50.0
PAD_S = 0.1
PEAK = 0.98
_ANSWER = re.compile(r"^(\d{2}) pref=(1|2|same|none) bad=(-|1|2|1,2)$")


def _read(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as clip:
        if clip.getnchannels() != 1 or clip.getsampwidth() != 2:
            raise ValueError(f"{path} is not 16-bit mono")
        samples = np.frombuffer(clip.readframes(clip.getnframes()), dtype="<i2").astype(np.float32) / 32767
        return samples, clip.getframerate()


def _write(path: Path, samples: np.ndarray, rate: int) -> None:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    path.write_bytes(buffer.getvalue())


def loudness_match(samples: np.ndarray, rate: int) -> tuple[np.ndarray, float]:
    """Trim lead and tail quiet to PAD_S and scale the speech frames' RMS to TARGET_DBFS (peak kept under
    PEAK), so loudness and silence length do not tell the models apart. Returns the clip and its gain in dB.
    """
    frame = max(1, int(0.02 * rate))
    count = len(samples) // frame
    if count == 0:
        raise ValueError("the clip is shorter than one frame")
    frames = samples[: count * frame].reshape(count, frame)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    active = np.flatnonzero(20 * np.log10(np.maximum(rms, 1e-12)) > FLOOR_DBFS)
    if active.size == 0:
        raise ValueError("the clip is silent")
    pad = int(PAD_S * rate)
    start = max(0, active[0] * frame - pad)
    end = min(len(samples), (active[-1] + 1) * frame + pad)
    level = 20 * np.log10(np.sqrt(np.mean(frames[active] ** 2)))
    gain = 10 ** ((TARGET_DBFS - level) / 20)
    clip = samples[start:end] * gain
    peak = float(np.abs(clip).max())
    if peak > PEAK:
        gain *= PEAK / peak
        clip *= PEAK / peak
    return clip.astype(np.float32), round(float(20 * np.log10(gain)), 2)


PAGE = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>음성 합성 듣기 비교</title>
<style>
:root {{ --bg: #fbfbfa; --fg: #1d1d1f; --muted: #5f6368; --line: #d9d9d6; --card: #ffffff;
  --accent: #2457c5; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg: #17181a; --fg: #ececec; --muted: #a3a6ab; --line: #34363a; --card: #1f2023;
    --accent: #8fb0ff; }}
}}
body {{ margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.6 system-ui, sans-serif; }}
main {{ max-width: 760px; margin: 0 auto; padding: 24px 16px 64px; }}
h1 {{ font-size: 1.4rem; margin: 0 0 8px; }}
p.note {{ color: var(--muted); margin: 0 0 20px; }}
section {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px;
  padding: 14px 16px; margin: 0 0 14px; }}
h2 {{ font-size: 1rem; margin: 0 0 6px; }}
.text {{ margin: 0 0 10px; }}
.players {{ display: grid; grid-template-columns: 1fr; gap: 6px; margin-bottom: 8px; }}
.players label {{ display: flex; gap: 8px; align-items: center; }}
audio {{ width: 100%; }}
fieldset {{ border: 0; padding: 0; margin: 4px 0; display: flex; flex-wrap: wrap; gap: 4px 16px; }}
legend {{ font-weight: 600; margin-bottom: 2px; }}
textarea {{ width: 100%; box-sizing: border-box; min-height: 90px; font: inherit; background: var(--card);
  color: var(--fg); border: 1px solid var(--line); border-radius: 6px; padding: 8px; }}
button {{ font: inherit; padding: 8px 14px; border-radius: 6px; border: 1px solid var(--accent);
  background: var(--accent); color: var(--bg); cursor: pointer; }}
</style>
</head>
<body>
<main>
<h1>음성 합성 듣기 비교 ({count}쌍)</h1>
<p class="note">쌍마다 같은 문장을 두 음성 합성이 읽었습니다. 어느 쪽이 어떤 모델인지는 쌍마다 무작위입니다.
1과 2를 모두 듣고 더 자연스러운 쪽을 고르세요. 서비스에 쓰기 싫을 만큼 부자연스러운 클립이 있으면 표시하세요.
끝나면 맨 아래 "결과 만들기"를 누르고 나온 글을 복사해 전달하거나 이 폴더에 answers.txt로 저장하세요.</p>
{pairs}
<section>
<h2>전체 의견 (선택)</h2>
<textarea id="comment" placeholder="느낀 점을 자유롭게 적어 주세요"></textarea>
</section>
<p><button type="button" id="make">결과 만들기</button></p>
<textarea id="result" readonly aria-label="결과"></textarea>
</main>
<script>
const COUNT = {count};
const KEY = "tts-ab-answers";
const form = () => document.querySelectorAll("input");
function save() {{
  const state = {{}};
  form().forEach((input) => {{ if (input.checked) state[input.id] = true; }});
  state.comment = document.getElementById("comment").value;
  try {{ localStorage.setItem(KEY, JSON.stringify(state)); }} catch (e) {{}}
}}
function restore() {{
  let state = {{}};
  try {{ state = JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{}}
  form().forEach((input) => {{ if (state[input.id]) input.checked = true; }});
  if (state.comment) document.getElementById("comment").value = state.comment;
}}
document.addEventListener("change", save);
document.getElementById("comment").addEventListener("input", save);
document.getElementById("make").addEventListener("click", () => {{
  const lines = [];
  for (let n = 1; n <= COUNT; n++) {{
    const id = String(n).padStart(2, "0");
    const pref = document.querySelector(`input[name="pref-${{id}}"]:checked`);
    const bad = ["1", "2"].filter((side) => document.getElementById(`bad-${{id}}-${{side}}`).checked);
    lines.push(`${{id}} pref=${{pref ? pref.value : "none"}} bad=${{bad.length ? bad.join(",") : "-"}}`);
  }}
  const comment = document.getElementById("comment").value.replace(/\\s+/g, " ").trim();
  if (comment) lines.push(`comment: ${{comment}}`);
  const result = document.getElementById("result");
  result.value = lines.join("\\n") + "\\n";
  result.focus();
  result.select();
}});
restore();
</script>
</body>
</html>
"""

PAIR = """<section>
<h2>{number}번</h2>
<p class="text">{text}</p>
<div class="players">
<label>1 <audio controls preload="none" src="clips/{number}-1.wav"></audio></label>
<label>2 <audio controls preload="none" src="clips/{number}-2.wav"></audio></label>
</div>
<fieldset><legend>더 자연스러운 쪽</legend>
<label><input type="radio" name="pref-{number}" id="pref-{number}-1" value="1"> 1</label>
<label><input type="radio" name="pref-{number}" id="pref-{number}-2" value="2"> 2</label>
<label><input type="radio" name="pref-{number}" id="pref-{number}-same" value="same"> 비슷함</label>
</fieldset>
<fieldset><legend>쓰기 싫을 만큼 부자연스러움</legend>
<label><input type="checkbox" id="bad-{number}-1"> 1</label>
<label><input type="checkbox" id="bad-{number}-2"> 2</label>
</fieldset>
</section>"""


def build(
    audio_root: Path,
    tag: str,
    a: str,
    b: str,
    ids,
    texts: dict[int, str],
    out: Path,
    key_path: Path,
    seed: int = SEED,
) -> dict:
    """Write the listening folder and the answer key; the key never goes into the folder."""
    out, key_path = Path(out), Path(key_path)
    if key_path.resolve().is_relative_to(out.resolve()):
        raise ValueError("the answer key must be outside the listening folder")
    rng = random.Random(seed)
    order = list(ids)
    rng.shuffle(order)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    pairs, sections = [], []
    for number, sentence in enumerate(order, 1):
        sides = [a, b]
        rng.shuffle(sides)
        entry: dict = {"pair": number, "id": sentence, "1": sides[0], "2": sides[1], "gain_db": {}}
        rates = set()
        for side, candidate in zip(("1", "2"), sides, strict=True):
            samples, rate = _read(Path(audio_root) / tag / candidate / "ko" / f"{sentence}.wav")
            clip, gain = loudness_match(samples, rate)
            _write(out / "clips" / f"{number:02d}-{side}.wav", clip, rate)
            entry["gain_db"][side] = gain
            rates.add(rate)
        if len(rates) != 1:
            raise ValueError(f"sentence {sentence}: the two clips have different sample rates {rates}")
        pairs.append(entry)
        sections.append(PAIR.format(number=f"{number:02d}", text=html.escape(texts[sentence])))
    (out / "index.html").write_text(
        PAGE.format(count=len(order), pairs="\n".join(sections)), encoding="utf-8"
    )
    key = {"a": a, "b": b, "tag": tag, "seed": seed, "built": datetime.now(UTC).isoformat(), "pairs": pairs}
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    return key


def unblind(key: dict, answers: str) -> dict:
    by_number = {f"{pair['pair']:02d}": pair for pair in key["pairs"]}
    models = (key["a"], key["b"])
    summary: dict = {
        "preferred": dict.fromkeys(models, 0),
        "same": 0,
        "unanswered": 0,
        "unacceptable": dict.fromkeys(models, 0),
        "comment": "",
        "pairs": [],
    }
    seen = set()
    for line in (line.strip() for line in answers.splitlines()):
        if not line:
            continue
        if line.startswith("comment:"):
            summary["comment"] = line.removeprefix("comment:").strip()
            continue
        match = _ANSWER.match(line)
        if not match or match.group(1) not in by_number:
            raise ValueError(f"cannot read the answer line for pair {line[:2]}: {line!r}")
        number, preference, bad = match.groups()
        pair = by_number[number]
        seen.add(number)
        if preference in ("1", "2"):
            summary["preferred"][pair[preference]] += 1
        else:
            summary["same" if preference == "same" else "unanswered"] += 1
        flagged = [] if bad == "-" else [pair[side] for side in bad.split(",")]
        for model in flagged:
            summary["unacceptable"][model] += 1
        summary["pairs"].append(
            {
                "pair": pair["pair"],
                "id": pair["id"],
                "preferred": pair.get(preference),
                "unacceptable": flagged,
            }
        )
    summary["unanswered"] += len(set(by_number) - seen)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    phases = parser.add_subparsers(dest="phase", required=True)
    build_parser = phases.add_parser("build")
    build_parser.add_argument("--tag", required=True)
    build_parser.add_argument("--a", required=True, help="one candidate, e.g. melo")
    build_parser.add_argument("--b", required=True, help="the other candidate")
    build_parser.add_argument("--out", type=Path, required=True)
    build_parser.add_argument("--key", type=Path, required=True)
    build_parser.add_argument("--sentences-dir", type=Path, help="export JSON instead of the FLEURS parquet")
    unblind_parser = phases.add_parser("unblind")
    unblind_parser.add_argument("--key", type=Path, required=True)
    unblind_parser.add_argument("--answers", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "build":
        from eval.tts_eval import AUDIO, sentences

        texts = {
            row["id"]: row["raw_transcription"]
            for row in sentences("ko", "validation", None, args.sentences_dir)
        }
        build(AUDIO, args.tag, args.a, args.b, AB_IDS, texts, args.out, args.key)
        print(f"written to {args.out / 'index.html'} (key: {args.key})")
    else:
        key = json.loads(args.key.read_text(encoding="utf-8"))
        summary = unblind(key, args.answers.read_text(encoding="utf-8"))
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        path = REPORTS / f"tts_ab_{key['tag']}_{stamp}.json"
        path.write_text(
            json.dumps({"key": key, "summary": summary}, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(json.dumps({k: v for k, v in summary.items() if k != "pairs"}, ensure_ascii=False))
        print(f"written to {path}")


if __name__ == "__main__":
    main()

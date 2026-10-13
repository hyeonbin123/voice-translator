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

# [파일 역할] 한국어 합성 후보와 MeloTTS를 사람이 블라인드로 듣고 비교하게 하는 도구다(T78 12절 3단계, T82
# 14절 3단계). build는 두 후보의 validation 음성으로 청취 폴더(번호만 붙은 클립과 index.html)와 그 폴더 밖의
# 답 열쇠 파일을 만들고, unblind는 청취자가 돌려준 결과 글을 열쇠로 풀어 모델별 선호 수와 '부자연' 표시 수를
# 센다.
# [왜 필요한가] 재인식 CER은 자연스러움을 대신하지 못한다(3절 한계). 그래서 후보가 자동 관문을 모두 넘으면
# 사람이 20문장을 블라인드로 듣고 거부하지 않을 때만 채택하도록 규칙을 정했다. 거부 여부는 듣는 사람이 정하며
# 수치 기준이 없다.
# [블라인드 설계] 쌍 순서와 쌍 안의 좌우를 시드로 섞고, 파일 이름에는 번호만 쓰며, 열쇠는 청취 폴더 밖에만
# 둔다. 두 클립 모두 앞뒤 조용한 부분과 음량을 같은 방법으로 맞춰 길이·크기로 모델을 알아채지 못하게 한다.
# [실제 사용] T78은 0단계, T82는 1단계에서 후보가 떨어져 이 청취 단계까지 가지 않았다(12·14절). 도구는
# tests/test_tts_eval.py가 검증한다.

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

# numpy: 프레임 RMS 계산과 음량 조절. 합성 음성 경로(AUDIO)와 문장 읽기는 main에서 eval.tts_eval을 불러 쓴다.
import numpy as np

from eval.common import REPORTS

# 청취할 20문장 ID. 합성하기 전에 정해 docs/experiments.md 12절에 같은 목록을 적었다. 결과를 보고 문장을
# 고르지 못하게 하려는 사전 고정이다.
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
# SEED: 쌍 순서와 좌우 섞기 시드(12절). TARGET_DBFS: 말소리 프레임 RMS를 맞출 목표 −23dBFS. FLOOR_DBFS: 이보다
# 조용한 20ms 프레임은 말소리가 아닌 것으로 본다. PAD_S: 앞뒤에 남길 조용한 부분 0.1초. PEAK: 키운 뒤 봉우리가
# 넘지 않을 상한(찌그러짐 방지).
# _ANSWER: 청취 페이지가 만드는 결과 한 줄 형식(쌍 번호 두 자리, 선호 1·2·same·none, 부자연 표시 -·1·2·1,2).
SEED = 7802  # pair order and the side of each model
TARGET_DBFS = -23.0  # RMS of the frames above the floor, after trimming
FLOOR_DBFS = -50.0
PAD_S = 0.1
PEAK = 0.98
_ANSWER = re.compile(r"^(\d{2}) pref=(1|2|same|none) bad=(-|1|2|1,2)$")


# 16비트 모노 WAV를 -1~1 float 배열과 표본화율로 읽는다. 다른 형식이면 비교 조건이 달라지므로 ValueError로
# 멈춘다.
def _read(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as clip:
        if clip.getnchannels() != 1 or clip.getsampwidth() != 2:
            raise ValueError(f"{path} is not 16-bit mono")
        samples = np.frombuffer(clip.readframes(clip.getnframes()), dtype="<i2").astype(np.float32) / 32767
        return samples, clip.getframerate()


# float 배열을 16비트 모노 WAV로 쓴다(범위를 넘는 값은 잘라 냄).
def _write(path: Path, samples: np.ndarray, rate: int) -> None:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    path.write_bytes(buffer.getvalue())


# 앞뒤 조용한 부분을 0.1초만 남기고 자르고, 말소리 프레임의 RMS를 −23dBFS로 맞춘다(12절 블라인드 A/B 규칙).
# 입력: 파형, 표본화율. 출력: 맞춘 클립과 적용한 이득(dB, 열쇠에 기록).
# 말소리 프레임만으로 음량을 재는 것은 문장 사이 쉼의 길이가 평균 음량을 바꾸지 않게 하려는 것이다. 키운 뒤
# 봉우리가 0.98을 넘으면 이득을 그만큼 줄인다.
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
    # −50dBFS보다 큰 20ms 프레임 = 말소리 프레임. 0으로 나누기·log 0을 막으려고 RMS 아래쪽을 1e-12로 자른다.
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


# 청취 페이지 틀(str.format으로 채움). 중괄호를 두 겹으로 쓴 것은 format이 CSS·JS의 중괄호를 자리표시자로 읽지
# 않게 하려는 것이다.
# 페이지 구성: 안내문, 쌍마다 문장·두 플레이어·'더 자연스러운 쪽'·'부자연' 표시, 전체 의견, '결과 만들기'
# 단추와 결과 글 상자. 선택 상태는 localStorage에 저장해 페이지를 다시 열어도 이어서 할 수 있다(저장이 막힌
# 브라우저에서도 동작하도록 try로 감쌈). 결과 글은 _ANSWER 형식의 줄들이다. 어두운 화면 설정이면 색 변수를
# 바꾼다.
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

# 쌍 하나의 HTML 틀. 클립 파일 이름은 '번호-좌우'뿐이라 모델 이름이 드러나지 않는다. preload='none'으로 누르기
# 전에는 음성을 받지 않는다.
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


# 청취 폴더와 답 열쇠를 만들고 열쇠 dict를 돌려준다. 입력: 합성 음성 폴더(data/tts_audio), 실행 태그, 두 후보
# 이름, 문장 ID, ID → 원문, 청취 폴더, 열쇠 경로, 시드.
# 열쇠 경로가 청취 폴더 안이면 청취자가 답을 볼 수 있으므로 거부한다. 쌍마다 두 클립의 표본화율이 다르면 소리
# 차이로 모델이 드러나므로 멈춘다.
# 열쇠에는 쌍 번호, 문장 ID, 좌우의 모델, 적용한 이득이 들어간다.
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


# 청취 결과 글을 열쇠로 풀어 요약한다: 모델별 선호 수, 비슷함 수, 답하지 않은 쌍 수(none과 결과에 없는 쌍),
# 모델별 부자연 표시 수, 전체 의견, 쌍별 풀이.
# 형식에 맞지 않거나 없는 쌍 번호가 있으면 ValueError로 멈춘다. 잘못 읽은 줄을 조용히 건너뛰면 집계가 틀어지기
# 때문이다.
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


# 명령줄 진입점. build는 FLEURS 한국어 validation 원문(또는 --sentences-dir로 내보낸 JSON)을 읽어 청취 폴더를
# 만들고, unblind는 요약을 reports/tts_ab_<tag>_*.json에 쓰고 쌍별 풀이를 뺀 요약을 출력한다.
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

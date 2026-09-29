"""The functions behind the recorded end-of-speech and live subtitle numbers (T33, T34, T61), on hand-made
cases. They need only numpy, so they run in CI without the eval group."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from eval.eos_eval import S_CHUNK, S_START_FRAMES, group, score
from eval.stream_eval import appearance_lags, dark_changes, erased

BACKEND = Path(__file__).resolve().parents[1]


def test_the_metrics_import_without_the_eval_group():
    # CI installs only the default groups: pyarrow is needed for reading FLEURS, not for the metrics.
    code = (
        "import sys; sys.modules['pyarrow'] = None; sys.modules['pyarrow.parquet'] = None; "
        "import eval.eos_eval, eval.stream_eval"
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=BACKEND, env=env, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr


def silero_group(flags: list[bool]) -> list[tuple[int, int, int]]:
    """group() with the Silero frames (32 ms) and the chosen 1000 ms of quiet, in frames."""
    return [
        (u["start"] // S_CHUNK, u["end"] // S_CHUNK, u["decided"] // S_CHUNK)
        for u in group(np.array(flags), S_CHUNK, 1000, S_START_FRAMES)
    ]


def test_an_utterance_ends_after_the_quiet_and_is_timed_from_its_first_speech_frame():
    # 1000 ms is 31 frames of 32 ms: the end is decided on the 31st quiet frame.
    assert silero_group([False] * 10 + [True] * 20 + [False] * 40) == [(10, 30, 61)]


def test_speech_shorter_than_250_ms_is_dropped():
    assert silero_group([False] * 10 + [True] * 7 + [False] * 40) == []  # 7 frames = 224 ms


def test_the_29_second_limit_splits_long_speech():
    utterances = silero_group([False] * 40 + [True] * 2000 + [False] * 40)
    # 906 frames with the 6 pre-roll frames: the split is decided at once, and speech goes on in a new clip.
    assert utterances[0] == (40, 940, 940)
    assert utterances[1][0] == 940


def test_split_merged_missed_and_false_utterances_and_the_latency():
    refs = [(0, 1000), (2000, 3000), (4000, 5000), (6000, 7000), (8000, 9000)]
    utterances = [
        {"start": 0, "end": 1000, "decided": 17_000},  # the first alone: 1 s after its end
        {"start": 2000, "end": 2500, "decided": 2600},  # the second in two parts
        {"start": 2500, "end": 3000, "decided": 3100},
        {"start": 4000, "end": 7000, "decided": 7100},  # the third and fourth as one
        {"start": 10_000, "end": 11_000, "decided": 11_100},  # nothing was said here
    ]  # and the fifth is missed
    assert score(utterances, refs) == {
        "utterances": 5,
        "split": 1,
        "missed": 1,
        "merged_pairs": 1,
        "pairs": 4,
        "false": 1,
        "latencies_ms": [1000.0],
    }


def test_a_changed_dark_word_counts_its_changed_letters_not_the_rest():
    # docs/experiments.md 8: "Japanese" heard again as "Javanese" changes one dark letter, not every dark
    # letter after it; erasure still counts what was taken off the end of the screen.
    shown = [("thejapanesepeople", 17), ("thejavanesepeoplelive", 17)]
    final = "thejavanesepeoplelive"
    assert dark_changes(shown, final) == (1, 1)
    assert erased(["thejapanese", "thejapanesepeople", "thejavanesepeople", final]) == 12


def test_dark_letters_the_final_result_changes_are_counted():
    assert dark_changes([("mouldis", 5)], "moldis") == (1, 0)


def test_a_word_appears_when_it_is_on_screen_to_stay():
    events = [(0.5, "hello"), (1.0, "help"), (1.5, "hellowor"), (2.0, "helloworld")]
    spans = [(0, 5, 0.3), (5, 10, 0.9)]
    # "hello" went away at 1.0 and came back at 1.5; "world" was shown whole only by the final result.
    assert appearance_lags(events, spans) == pytest.approx([1.2, 1.1])

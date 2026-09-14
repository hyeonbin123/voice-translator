"""What the live service check sends for a clip, as the browser would (T61, T62). No models, no GPU."""

import numpy as np
import pytest

pytest.importorskip("pyarrow")  # eval.eos_eval reads FLEURS parquet files; the eval group installs it

from eval.stream_eval import PAD_FRAMES, QUIET_END_FRAMES, browser_plan  # noqa: E402

# Six quiet frames, speech, an inner pause of eight quiet frames, speech, and the six quiet frames that end
# the clip conversation mode would send. Speech starts at frame 6; the browser starts streaming at frame 8.
FLAGS = np.array([False] * 6 + [True] * 10 + [False] * 8 + [True] * 5 + [False] * 6)
DETECTED = 8
INNER_QUIET = 8


@pytest.mark.parametrize("pause_frames", [6, 12, 19])
def test_the_final_is_asked_for_after_the_chosen_quiet_and_before_the_end(pause_frames):
    plan = browser_plan(FLAGS, DETECTED, pause_frames)
    end_at = len(FLAGS) - PAD_FRAMES + QUIET_END_FRAMES
    assert plan[-1] == (end_at, "end", None)
    pauses = [at for at, kind, _ in plan if kind == "pause"]
    assert pauses[-1] == len(FLAGS) - PAD_FRAMES + pause_frames < end_at
    inner = INNER_QUIET >= pause_frames  # only a long enough inner pause asks for a final, then resumes
    kinds = [kind for _, kind, _ in plan]
    assert kinds.count("pause") == 1 + inner and kinds.count("resume") == inner
    assert kinds[:2] == ["utterance", "audio"] and plan[0][0] == DETECTED


@pytest.mark.parametrize("pause_frames", [6, 12, 19])
def test_at_the_last_pause_the_server_has_exactly_the_clip(pause_frames):
    plan = browser_plan(FLAGS, DETECTED, pause_frames)
    last_pause = max(i for i, (_, kind, _) in enumerate(plan) if kind == "pause")
    sent = [frame for _, kind, span in plan[:last_pause] if kind == "audio" for frame in range(*span)]
    # Every frame of the clip once, in order: the WAV conversation mode would upload.
    assert sent == list(range(len(FLAGS)))
    assert not any(kind == "audio" for _, kind, _ in plan[last_pause:])

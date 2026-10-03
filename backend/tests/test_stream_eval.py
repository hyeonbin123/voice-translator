"""What the live service check sends for a clip, as the browser would (T61, T62). No models, no GPU."""

import numpy as np
import pytest

from app.services.live import wav_from_pcm
from eval.eos_eval import RATE, S_CHUNK
from eval.stream_eval import (
    PAD_FRAMES,
    QUIET_END_FRAMES,
    browser_plan,
    classify,
    clip_wav,
    decide,
    end_frame,
    paired_ci,
    quantize,
    summarize_modes,
    turns,
)

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


# T77 (docs/experiments.md 11): the two paths per clip and the rule, without a server.


def clip_and_plan():
    rng = np.random.default_rng(7)
    audio = quantize(rng.uniform(-0.5, 0.5, len(FLAGS) * S_CHUNK).astype(np.float32))
    utterance = {"audio": audio, "flags": FLAGS, "detected_s": DETECTED * S_CHUNK / RATE}
    return utterance, browser_plan(FLAGS, DETECTED, PAD_FRAMES)


def test_the_uploaded_wav_is_the_file_the_server_builds_from_the_stream_at_the_last_pause():
    utterance, plan = clip_and_plan()
    pcm = np.round(utterance["audio"] * 32768).astype("<i2")
    last_pause = max(i for i, (_, kind, _) in enumerate(plan) if kind == "pause")
    streamed = b"".join(
        pcm[span[0] * S_CHUNK : span[1] * S_CHUNK].tobytes()
        for _, kind, span in plan[:last_pause]
        if kind == "audio"
    )
    assert clip_wav(utterance["audio"]) == wav_from_pcm(streamed)
    # The upload goes out when the browser decides the end, as the streamed path's end message does.
    assert end_frame(utterance) == plan[-1][0] == len(FLAGS) - PAD_FRAMES + QUIET_END_FRAMES


def answer(text="안녕", lang="ko", guessed=None, audio=True):
    result = {
        "id": "x",
        "source_lang": lang,
        "target_lang": "en" if lang == "ko" else "ko",
        "source_text": text,
        "translated_text": f"[{text}]",
        "stt_model": "stt",
        "mt_model": "mt",
        "tts_model": "tts" if audio else None,
        "tts_error": None if audio else "Speech synthesis failed",
        "audio_id": "a" if audio else None,
    }
    if guessed is not None:
        result["language_guessed"] = guessed
    return {"outcome": "final", "result": result}


def test_a_clip_is_the_same_when_its_texts_languages_models_and_audio_presence_match():
    assert classify(answer(), answer(), []) == "same"
    # Record ids, times and audio bytes differ on every call and do not count.
    other = answer()
    other["result"] |= {"id": "y", "stt_ms": 5}
    other["audio_sha256"] = "different bytes"
    assert classify(answer(), other, []) == "same"
    error = {"outcome": "error", "detail": "No speech was recognized"}
    assert classify(error, dict(error), []) == "same"
    assert classify(answer(), answer(audio=False), []) != "same"
    assert classify(answer(guessed=False), answer(guessed=True), []) != "same"


def test_a_differing_clip_counts_against_the_prepare_only_when_uploads_repeat_the_first_answer():
    upload, prepared = answer("하나"), answer("둘")
    assert classify(upload, prepared, [answer("하나"), answer("하나")]) == "different"
    assert classify(upload, prepared, [answer("하나"), answer("둘")]) == "undecidable"
    assert classify(upload, prepared, [answer("셋"), answer("셋")]) == "undecidable"
    assert classify(upload, prepared, []) == "undecidable"


def test_the_paired_interval_is_seeded_and_holds_the_median():
    differences = [0.6, 0.7, 0.8, 0.65, 0.9, 0.75, 0.7, 0.72]
    median, low, high = paired_ci(differences)
    assert median == pytest.approx(0.71)
    assert low <= median <= high
    assert paired_ci(differences) == (median, low, high)


def row(mode, lang, upload_s, prepared_s, identity="same", resumes=0):
    return {
        "mode": mode,
        "lang": lang,
        "identity": identity,
        "upload": {**answer(), "audio_s": upload_s, "final_s": upload_s - 0.1},
        "prepared": {
            **answer(),
            "audio_s": prepared_s,
            "final_s": prepared_s - 0.1,
            "pauses": 1 + resumes,
            "resumes": resumes,
        },
    }


def table(saving_en=0.7, saving_ko=0.7, identity="same", count=3, modes=("conversation",)):
    rows = []
    for mode in modes:
        for lang, saving in (("en", saving_en), ("ko", saving_ko)):
            rows += [row(mode, lang, 2.0 + i / 10, 2.0 + i / 10 - saving, resumes=i) for i in range(count)]
    if identity != "same":
        rows[0]["identity"] = identity
    return summarize_modes(rows)


def test_the_summary_counts_prepares_dropped_prepares_and_the_saving():
    [en, ko] = table()
    assert (en["mode"], en["lang"], ko["lang"]) == ("conversation", "en", "ko")
    assert en["saving_ms"] == pytest.approx(700) and en["paired_median_ms"] == pytest.approx(700)
    assert (en["upload_p50_ms"], en["prepared_p50_ms"]) == (pytest.approx(2100), pytest.approx(1400))
    assert (en["prepares"], en["dropped"]) == (6, 3)
    assert (en["same"], en["different"], en["undecidable"]) == (3, 0, 0)


def test_the_rule_adopts_a_mode_only_when_both_directions_save_300_ms_and_nothing_differs():
    assert decide(table())["conversation"] == {"adopt": True, "reasons": []}
    assert decide(table(saving_ko=0.299))["conversation"]["adopt"] is False
    assert decide(table(saving_en=0.300))["conversation"]["adopt"] is True
    assert decide(table(identity="different"))["conversation"]["adopt"] is False
    assert decide(table(identity="undecidable"))["conversation"]["adopt"] is True
    many = table(count=4)
    many[0]["undecidable"] = 3
    assert decide(many)["conversation"]["adopt"] is False
    assert decide(table()[:1])["conversation"]["adopt"] is False  # one direction only
    both = decide(table(modes=("conversation", "dialog"), saving_ko=0.2))
    assert set(both) == {"conversation", "dialog"} and not any(v["adopt"] for v in both.values())


def test_dialog_clips_take_turns_between_the_languages_on_one_connection():
    chosen = {"en": [("e1", "u", "f"), ("e2", "u", "f")], "ko": [("k1", "u", "f")]}
    [(name, items)] = turns("dialog", chosen)
    assert name == "dialog" and [(key, lang) for key, lang, _, _ in items] == [
        ("e1", "en"),
        ("k1", "ko"),
        ("e2", "en"),
    ]
    assert [name for name, _ in turns("conversation", chosen)] == ["en", "ko"]

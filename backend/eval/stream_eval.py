"""T34: offline check of live subtitles while speaking (docs/experiments.md 8).

  uv run python -m eval.stream_eval run --split validation --tag t34_dev
  uv run python -m eval.stream_eval modes --split validation --tag t77_dev   (T77, docs/experiments.md 11)

Each FLEURS clip is cut as conversation mode cuts it in the browser (Silero VAD, 1000 ms of quiet, T33),
then streamed through a simulated clock. While the person speaks, the server recognizes all of the
utterance's audio received so far and translates the result, one update at a time on its single model
thread. The model calls are real and timed on this machine's GPU; their measured times advance the clock.
Network time is taken as zero. Once the browser has seen 192 ms of quiet, the clip conversation mode would
send is complete: the server recognizes it with the final settings, and that result replaces the live text.

Texts are compared by their letters only (app/services/live.py): Korean spacing changes between results.
"""

# [파일 역할] 동시통역 자막과 대화 모드·두 사람 대화의 '미리 처리'를 재는 측정 도구다. 하위 명령 세 가지:
# - run (T34, docs/experiments.md 8절): 오프라인 모의 측정. 실제 모델 호출 시간으로 가상의 시계를 진행하며
#   자막 갱신 간격(250·500·1000ms)과 갱신 때 인식 설정(P1·P5)의 조합을 비교한다. 여기서 1000ms·P5를 골랐다.
# - service (T58·T61·T63, 8절): docker compose 서비스에 nginx를 거쳐 브라우저처럼 32ms 조각을 실제 시간으로
#   흘려보내 자막 표시 지연과 말 끝 → 번역 음성 시간을 잰다. T82·T83의 서버 확인에도 같은 명령을 쓰도록 규칙에
#   적혀 있다.
# - modes (T77, 11절): 대화 모드·두 사람 대화를 옛 경로(1초 판정 뒤 HTTP 업로드)와 새 경로(웹소켓으로 흘리고
#   192ms 쉼에서 미리 처리)로 번갈아 재고, 두 경로의 결과가 같은지와 채택 규칙을 판정한다.
# [왜 필요한가] 자막·미리 처리는 시간에 따라 바뀌는 동작이라 문장 하나의 정답률로는 잴 수 없다. 브라우저가
# 소리를 자르고 보내는 방식(Silero VAD, 1000ms 쉼, 앞뒤 192ms)을 그대로 흉내 내야 서버가 받는 소리가 실제와
# 같아진다.
# [설계 이유] 글자 비교는 서버와 같은 app/services/live.py 함수(letters 등)를 써서 측정과 서비스가 같은
# 기준으로 '같다'를 판단한다. 소리 자르기는 T33 측정 도구(eval/eos_eval.py)의 VAD·데이터 읽기를 그대로
# 가져온다.

from __future__ import annotations

import argparse
import asyncio
import contextlib
import gc
import hashlib
import itertools
import json
import random
import secrets
import statistics
import time
import uuid
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path

# numpy: 파형·VAD 판정 배열과 백분위·붓스트랩 계산.
import numpy as np

# app.services.live: 서버의 동시통역 처리와 같은 글자 비교(letters: 대소문자·띄어쓰기·문장부호를 뺀 글자),
# 공통 앞부분 길이, 진하게 보일 길이(stable_length, LocalAgreement-2), PCM → WAV 변환.
# eval.eos_eval: T33에서 만든 Silero VAD 실행기, 판정 묶기(group), FLEURS 클립 읽기와 음량 맞춤, 분홍 잡음,
# 프레임 크기·시드 상수. 대화 모드의 말 끝 판정과 같은 계산이다.
from app.services.live import common_start, letters, stable_length, wav_from_pcm
from eval.common import DATA, MODELS, PROJECT, REPORTS, gpu_memory_mb
from eval.eos_eval import (
    LANGS,
    PREROLL_MS,
    QUIET_FLOOR_DBFS,
    RATE,
    S_CHUNK,
    S_START_FRAMES,
    SEED,
    Silero,
    group,
    load_clips,
    pink_noise,
)

# SILENCE_MS: 말 끝으로 보는 쉼(1000ms). 7절 T33에서 정한 말 끝 판정(Silero VAD + 1000ms).
# PAD_FRAMES: 말소리 앞뒤로 붙이는 200ms를 32ms 조각 수로 바꾼 값(6조각 = 192ms). 브라우저
# frontend/src/conversation/silero.ts의 PAD_FRAMES와 같은 계산이다. 말 끝 뒤 6조각 조용하면 대화 모드가 보낼
# 소리가 다 모이므로, 미리 처리를 시작하는 쉼도 이 값이다.
# MAX_CLIP_S: 브라우저가 29초에서 한 마디를 강제로 끊으므로 그보다 긴 클립은 둘로 잘린다. 그런 클립은 뺀다.
# LEAD_S, TAIL_S: FLEURS 음성 앞뒤에 붙이는 조용한 구간(앞 1초, 뒤 1.5초, 8절 시험 방법).
# PER_LANGUAGE: 언어별 문장 수 40(8절 데이터). INTERVALS_MS: 후보 갱신 간격 U.
SILENCE_MS = 1000  # D9
PAD_FRAMES = round(PREROLL_MS / (S_CHUNK * 1000 / RATE))  # 6 frames kept before and after speech (silero.ts)
MAX_CLIP_S = 28.9  # longer clips are cut in two by the browser's 29 s limit
LEAD_S, TAIL_S = 1.0, 1.5
PER_LANGUAGE = 40
INTERVALS_MS = (250, 500, 1000)
# 인식 설정. BASE는 서버와 같은 VAD 필터와 무음 판정 문턱(0.6), FINAL은 최종 인식(beam 5, temperature를 올려
# 다시 푸는 기본 동작). PARTIAL은 자막 갱신 후보: P1 = beam 1·temperature 0만(다시 풀지 않아 빠름), P5 =
# 최종과 같은 설정(8절 '고르는 것').
# Every setting keeps the server's VAD filter (config.stt_vad_filter) and no-speech threshold.
BASE = {"vad_filter": True, "no_speech_threshold": 0.6}
FINAL = {**BASE, "beam_size": 5}  # conversation mode's recognition (app/services/models.py)
PARTIAL = {"P1": {**BASE, "beam_size": 1, "temperature": 0.0}, "P5": FINAL}


# float 파형을 브라우저가 보내는 16비트 PCM 정밀도로 한 번 양자화했다가 float로 되돌린다. 서버가 받는 소리와
# 같은 값으로 인식하게 하려는 것이다.
def quantize(audio: np.ndarray) -> np.ndarray:
    """The 16-bit PCM the browser sends."""
    return (np.clip(audio, -1, 1) * 32767).astype(np.int16).astype(np.float32) / 32768


# 한 마디 안에서 PAD_FRAMES(192ms) 이상 이어진 조용한 구간의 수를 센다. 이런 쉼마다 서버는 미리 처리를
# 시작했다가, 다시 말이 이어지면 버린다('헛 미리 처리'). 보고서에 소리 1분당 횟수로 적는다(8절 '참고로만 적는
# 것').
# 말소리 조각을 만날 때마다 직전 조용한 길이가 기준 이상이었는지 더한다.
def long_pauses(flags: np.ndarray) -> int:
    """Quiet runs of at least PAD_FRAMES inside an utterance: each starts a final that speech then drops."""
    count = run = 0
    for flag in flags:
        if flag:
            count += run >= PAD_FRAMES
            run = 0
        else:
            run += 1
    return count


# FLEURS 녹음 하나를 대화 모드가 브라우저에서 자르듯 잘라, 서버로 보낼 클립을 돌려준다. 한 마디로 정확히
# 하나가 나오지 않으면(중간에 1초 넘게 쉼) 또는 29초를 넘으면 None.
# 출력: audio(16비트로 양자화한 클립), detected_s(클립 안에서 브라우저가 말 시작을 판정해 흘려보내기 시작하는
# 시각), pauses(헛 미리 처리를 부를 쉼 수), flags(클립 조각마다 브라우저의 말소리 판정).
def cut(silero: Silero, clip: np.ndarray, seed: int) -> dict | None:
    """The clip conversation mode would send for this recording, or None unless it is exactly one."""
    # 앞 1초·뒤 1.5초의 조용한 구간을 붙이고 −60dBFS 분홍 잡음 바닥을 깐다. 실제 마이크는 완전한 0을 내지 않기
    # 때문이다(7절 측정 세부). 잡음 시드는 클립마다 정해져 실행마다 같다.
    track = np.concatenate(
        [np.zeros(int(LEAD_S * RATE), np.float32), clip, np.zeros(int(TAIL_S * RATE), np.float32)]
    )
    noise = pink_noise(len(track), np.random.default_rng(seed)) * 10 ** (QUIET_FLOOR_DBFS / 20)
    track = (track + noise).astype(np.float32)  # the VAD takes float32 only
    # 32ms 조각마다 Silero 말소리 판정 → 1000ms 쉼 기준으로 마디로 묶는다(T33과 같은 규칙).
    flags = silero.flags(track)
    utterances = group(flags, S_CHUNK, SILENCE_MS, S_START_FRAMES)
    if len(utterances) != 1:
        return None
    # 마디의 첫·끝 조각 번호. 클립은 첫 말소리 앞 6조각(트랙 시작보다 앞이면 있는 만큼)부터 끝 뒤 6조각까지다.
    # 브라우저가 앞뒤 192ms를 붙여 보내는 규칙과 같다.
    first, end = utterances[0]["start"] // S_CHUNK, utterances[0]["end"] // S_CHUNK
    begin = first - min(PAD_FRAMES, first)
    if (end + PAD_FRAMES - begin) * S_CHUNK / RATE >= MAX_CLIP_S:
        return None
    return {
        "audio": quantize(track[begin * S_CHUNK : (end + PAD_FRAMES) * S_CHUNK]),
        "detected_s": (first + S_START_FRAMES - begin) * S_CHUNK / RATE,  # when the browser starts streaming
        "pauses": long_pauses(flags[first:end]),
        "flags": flags[begin : end + PAD_FRAMES],  # the browser's speech flags, one per frame of the clip
    }


# faster-whisper로 인식하고 (텍스트, 걸린 초)를 돌려준다. faster-whisper의 transcribe는 결과를 생성기로
# 돌려주고 실제 계산은 segment를 읽을 때 일어나므로, 글을 다 읽을 때까지를 시간에 넣는다.
def recognize(model, audio: np.ndarray, lang: str, options: dict) -> tuple[str, float]:
    began = time.perf_counter()
    segments, _ = model.transcribe(audio, language=lang, **options)
    text = " ".join(segment.text.strip() for segment in segments).strip()  # the model runs while this reads
    return text, time.perf_counter() - began


# 번역하고 (번역문, 걸린 초)를 돌려준다.
def translate(translator, text: str, lang: str, target: str) -> tuple[str, float]:
    began = time.perf_counter()
    translation = translator.translate(text, lang, target)
    return translation, time.perf_counter() - began


# 클립 전체에 대한 대화 모드의 최종 결과(최종 설정 인식 + 번역)와 단어별 끝 시각을 만든다. 결과가 없으면
# (None, 이유)를 돌려주고, 이유는 utterances가 뺀 문장 수를 세는 데 쓴다.
# 단어 끝 시각은 표시 지연(단어가 끝난 뒤 자막에 나타나기까지)과 말 끝(마지막 단어 끝)의 기준이 된다.
def final_result(model, translator, audio: np.ndarray, lang: str, target: str) -> tuple[dict | None, str]:
    """Conversation mode's result for the clip, with each word's letters and end time, or why there is
    none."""
    text, stt_s = recognize(model, audio, lang, FINAL)
    if not letters(text):
        return None, "no_speech"
    # A second, untimed pass for word times; it must have the timed pass's letters.
    segments, _ = model.transcribe(audio, language=lang, word_timestamps=True, **FINAL)
    timed = [word for segment in segments for word in (segment.words or [])]
    # 단어 시각을 얻으려고 한 두 번째 인식이 시간을 잰 첫 인식과 글자가 다르면 단어 위치를 맞출 수 없으므로 그
    # 문장을 뺀다.
    if letters("".join(word.word for word in timed)) != letters(text):
        return None, "word_times_differ"
    # 단어마다 (최종 글자열에서의 시작, 끝 위치, 소리에서 끝난 시각)을 만든다. 위치는 letters 기준이라
    # 띄어쓰기·문장부호 차이와 무관하다.
    spans, cursor = [], 0
    for word in timed:
        size = len(letters(word.word))
        if size:  # a word of punctuation alone has no letters to show
            spans.append((cursor, cursor + size, word.end))
        cursor += size
    translation, mt_s = translate(translator, text, lang, target)
    return {"text": text, "spans": spans, "stt_s": stt_s, "translation": translation, "mt_s": mt_s}, ""


# 화면 글(shown)과 최종 글(final)을 difflib으로 정렬해, 최종 글의 각 글자가 화면 글에 있는지 표시한 배열을
# 돌려준다. 앞글자 비교가 아니라 정렬이라, 중간 단어 하나가 바뀌어도 그 뒤 글자는 '있음'으로 남는다.
# autojunk=False는 긴 글에서 자주 나오는 글자를 무시하는 difflib의 기본 동작을 꺼 정렬이 빠지지 않게 한다.
def covered(shown: str, final: str) -> np.ndarray:
    """Which letters of the final text the shown text has, by aligning the two."""
    mask = np.zeros(len(final), bool)
    for _, start, size in SequenceMatcher(None, shown, final, autojunk=False).get_matching_blocks():
        mask[start : start + size] = True
    return mask


# 표시 지연: 최종 단어마다 '그 단어의 글자가 화면에 모두 나타나 이후 계속 남아 있게 된 첫 시각 − 소리에서 그
# 단어가 끝난 시각'(8절 지표). 입력 events는 (보인 시각, 글자열) 목록이고 마지막이 최종 결과다.
# 뒤에서부터 거꾸로 보며 단어가 계속 덮여 있던 가장 이른 시각을 찾는다. 최종 결과에서야 처음 맞춰진 단어는
# 최종 결과가 보인 시각이 된다. 호출 위치: simulate(오프라인), stream_one(실제 서비스).
def appearance_lags(events: list[tuple[float, str]], spans: list[tuple[int, int, float]]) -> list[float]:
    """Per final word: the first time its letters were all on screen and stayed, minus the word's end.
    Events are (time, letters) with the final result last."""
    final = events[-1][1]
    masks = [covered(shown, final) for _, shown in events]
    lags = []
    for start, end, word_end in spans:
        appeared = events[-1][0]
        for (shown_at, _), mask in zip(reversed(events), reversed(masks), strict=True):
            if not mask[start:end].all():
                break
            appeared = shown_at
        lags.append(appeared - word_end)
    return lags


# 흔들림(normalized erasure)의 분자: 갱신마다 화면 글 끝에서 지워진 글자 수(이전 글 길이 − 공통 앞부분)의 합.
# 이 값을 최종 글자 수로 나눈 것이 흔들림이다(8절, Arivazhagan 외 2020을 글자 단위로).
def erased(sequence: list[str]) -> int:
    """Letters taken off the end of the screen from one update to the next (normalized erasure's
    numerator)."""
    total, before = 0, ""
    for shown in sequence:
        total += len(before) - common_start(before, shown)
        before = shown
    return total


# 확정 뒤 바뀜: (진하게 보인 뒤 다음 글에서 정렬로 찾을 수 없게 된 글자 수, 새로 진하게 된 글자 수). 비율은
# summarize가 전체 합으로 낸다.
# 처음에는 앞글자 기준으로 셌는데, 진한 부분에서 단어 하나만 바뀌어도 그 뒤 진한 글자가 모두 지워진 것으로
# 세어져 뜻보다 크게 나왔다. 그래서 첫 validation 결과를 본 뒤 글자 정렬 기준으로 정의를 고치고 validation을
# 다시 쟀다(8절 '지표 수정').
def dark_changes(shown: list[tuple[str, int]], final: str) -> tuple[int, int]:
    """(dark letters later changed, letters that turned dark) over (letters, dark letters) on screen.

    A dark letter has changed when the next text no longer has it, by aligning the two texts (the rule
    fixed after the first validation, docs/experiments.md 8: a word changed early in the dark part no
    longer counts every dark letter after it). The final result can change dark letters too; its own
    letters are not counted as turning dark."""

    # 이전 글의 진한 앞부분(dark 글자) 가운데 다음 글에서 찾을 수 없는 글자 수.
    def lost(before: str, dark: int, after: str) -> int:
        return int((~covered(after, before)[:dark]).sum())

    dropped = became = 0
    for (before, dark_before), (after, dark_after) in zip(shown, shown[1:], strict=False):
        changed = lost(before, dark_before, after)
        dropped += changed
        # 새로 진하게 된 글자 = 다음 글의 진한 글자 수 − (이전 진한 글자 중 살아남은 수). 음수가 되지 않게
        # 0에서 자른다.
        became += max(0, dark_after - (dark_before - changed))
    # 최종 결과도 진한 글자를 바꿀 수 있으므로 마지막 화면 글과 최종 글을 한 번 더 견준다. 최종 글 자신의
    # 글자는 '진하게 됨'으로 세지 않는다.
    if shown:
        dropped += lost(*shown[-1], final)
    return dropped, became


# 한 클립의 자막 갱신을 가상의 시계로 흉내 낸다(오프라인 run 전용). 입력: 모델, 번역기, cut의 클립,
# final_result의 최종 결과, 언어, 갱신 간격(초), 갱신 인식 설정. 출력: 클립 하나의 지표 dict.
# 서버처럼 모델 스레드가 하나라 갱신은 한 번에 하나만 돌고(free_at까지 기다림), 끝나면 그때까지 받은 소리로
# 다음 갱신을 한다. 실제 모델 호출 시간만큼 시계를 진행하고 네트워크 시간은 0으로 둔다(8절 시험 방법).
def simulate(
    model, translator, utterance: dict, final: dict, lang: str, target: str, interval_s: float, options: dict
) -> dict:
    audio = utterance["audio"]
    duration = len(audio) / RATE
    free_at, last_start = 0.0, None  # when the model thread is free, when the last update started
    previous = previous_translation = None
    sources: list[tuple[float, str, int]] = []  # (shown at, text, dark characters)
    translations: list[tuple[float, str, int]] = []
    model_s, updates = 0.0, 0
    while True:
        # 다음 갱신은 (브라우저가 흘려보내기 시작한 뒤, 모델 스레드가 비었을 때, 앞 갱신 시작 + 간격 뒤) 중
        # 가장 늦은 때 시작한다. 그 시각이 클립 끝을 넘으면 소리가 다 모인 것이라 최종 처리로 넘어간다.
        earliest = interval_s if last_start is None else last_start + interval_s
        start = max(utterance["detected_s"], free_at, earliest)
        if start >= duration:  # the whole clip is in: the final takes over
            break
        # 갱신 = 그 시각까지 받은 소리 전체를 다시 인식. 진한 길이는 바로 앞 결과와 같은 앞부분을 단어
        # 경계까지 자른 것(LocalAgreement-2).
        text, took = recognize(model, audio[: int(start * RATE)], lang, options)
        model_s += took
        updates += 1
        last_start, free_at = start, start + took
        dark = stable_length(previous, text)
        previous = text
        # 글과 진한 길이가 직전과 같으면 화면에 보낼 것이 없다. 원문 글이 바뀌고 글자가 있을 때만 다시
        # 번역하며, 번역 시간도 모델 스레드를 차지하므로 free_at을 늘린다.
        if sources and (text, dark) == sources[-1][1:]:
            continue  # nothing new to show or translate
        text_changed = not sources or text != sources[-1][1]
        sources.append((free_at, text, dark))
        if text_changed and letters(text):
            translation, took = translate(translator, text, lang, target)
            model_s += took
            free_at += took
            translations.append((free_at, translation, stable_length(previous_translation, translation)))
            previous_translation = translation
    # 최종 처리는 클립이 다 모인 뒤, 진행 중인 갱신이 끝나기를 기다렸다가 시작한다. 최종 원문은 최종 인식 시간
    # 뒤, 최종 번역문은 그 뒤 번역 시간 뒤에 보인다.
    final_start = max(duration, free_at)  # an update still running holds the one model thread
    source_final_at = final_start + final["stt_s"]
    translation_final_at = source_final_at + final["mt_s"]
    final_letters, translation_letters = letters(final["text"]), letters(final["translation"])
    # 지표는 모두 letters(대소문자·띄어쓰기·문장부호를 뺀 글자)로 센다. 진한 길이도 원래 글의 앞 dark 글자를
    # letters로 바꾼 길이로 맞춘다.
    source_dark = [(letters(text), len(letters(text[:dark]))) for _, text, dark in sources]
    translation_dark = [(letters(text), len(letters(text[:dark]))) for _, text, dark in translations]
    events = [(at, shown) for (at, _, _), (shown, _) in zip(sources, source_dark, strict=True)]
    source_dropped, source_became = dark_changes(source_dark, final_letters)
    translation_dropped, translation_became = dark_changes(translation_dark, translation_letters)
    # final_wait_s: 앞 갱신 때문에 최종 처리가 늦게 시작된 시간. final_text_s: 마지막 단어 끝부터 최종
    # 번역문이 보이기까지(합성 제외).
    return {
        "lags": appearance_lags(events + [(source_final_at, final_letters)], final["spans"]),
        "letters": len(final_letters),
        "source_erased": erased([shown for shown, _ in source_dark] + [final_letters]),
        "source_dark_dropped": source_dropped,
        "source_dark_became": source_became,
        "translation_letters": len(translation_letters),
        "translation_erased": erased([shown for shown, _ in translation_dark] + [translation_letters]),
        "translation_dark_dropped": translation_dropped,
        "translation_dark_became": translation_became,
        "model_s": model_s,
        "duration_s": duration,
        "updates": updates,
        "final_wait_s": final_start - duration,
        "final_text_s": translation_final_at - final["spans"][-1][2],
    }


# 이 PC의 GPU에 서버와 같은 모델을 올린다: large-v3-turbo(float16), opus-mt-tc-big 두 방향(영→한은 문장 단위,
# T17). add_cuda_dll_dirs는 Windows에서 CTranslate2가 CUDA DLL을 찾게 하는 서버 코드다(1절 실행 환경 메모).
def load_models():
    from faster_whisper import WhisperModel

    from app.services.cuda import add_cuda_dll_dirs
    from app.services.translation import DirectionalTranslator, MarianTranslator

    add_cuda_dll_dirs()
    model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    ct2 = MODELS / "ct2"
    translator = DirectionalTranslator(
        {
            ("ko", "en"): MarianTranslator(ct2 / "opus-mt-tc-big-ko-en", "ko", "en"),
            ("en", "ko"): MarianTranslator(ct2 / "opus-mt-tc-big-en-ko", "en", "ko", by_sentence=True),
        }
    )
    return model, translator


# 언어별로 FLEURS 클립을 고정 시드로 섞은 순서에서 쓸 수 있는 앞의 count개를 고른다. 쓸 수 없는 클립은
# 이유별로 센다(한 마디가 아님, 말 없음, 단어 시각 불일치). 시드 문자열에 분할·언어를 넣어 실행마다 같은
# 문장이 골라지고, 클립마다 잡음 시드도 고정한다.
# 호출 위치: run, choose_clips(service), modes.
def utterances(silero: Silero, model, translator, lang: str, split: str, count: int) -> tuple[list, dict]:
    """The first `count` usable clips in a seeded order, with the number skipped for each reason."""
    target = "ko" if lang == "en" else "en"
    clips = load_clips(lang, split)
    random.Random(f"stream-{SEED[split]}-{lang}").shuffle(clips)
    chosen, skipped = [], {"not_one_utterance": 0, "no_speech": 0, "word_times_differ": 0}
    for index, (key, clip) in enumerate(clips):
        utterance = cut(silero, clip, seed=SEED[split] * 1000 + index)
        if utterance is None:
            skipped["not_one_utterance"] += 1
            continue
        final, reason = final_result(model, translator, utterance["audio"], lang, target)
        if final is None:
            skipped[reason] += 1
            continue
        chosen.append((key, utterance, final))
        if len(chosen) == count:
            break
    return chosen, skipped


# 클립별 지표를 한 조합의 요약으로 모은다. 지연은 모든 단어를 모아 p50·p90, 비율 지표(흔들림, 확정 뒤 바뀜,
# 모델 사용, 초당 갱신)는 클립별 비율의 평균이 아니라 분자 합 ÷ 분모 합이다.
def summarize(results: list[dict]) -> dict:
    def ratio(part: str, whole: str) -> float:
        total = sum(r[whole] for r in results)
        return sum(r[part] for r in results) / total if total else 0.0

    lags = [lag for r in results for lag in r["lags"]]
    waits = [r["final_wait_s"] for r in results]
    return {
        "utterances": len(results),
        "lag_p50_ms": statistics.median(lags) * 1000,
        "lag_p90_ms": float(np.percentile(lags, 90)) * 1000,
        "source_erasure": ratio("source_erased", "letters"),
        "source_dark_dropped": ratio("source_dark_dropped", "source_dark_became"),
        "translation_erasure": ratio("translation_erased", "translation_letters"),
        "translation_dark_dropped": ratio("translation_dark_dropped", "translation_dark_became"),
        "model_use": ratio("model_s", "duration_s"),
        "updates_per_s": ratio("updates", "duration_s"),
        "final_wait_p50_ms": statistics.median(waits) * 1000,
        "final_wait_p90_ms": float(np.percentile(waits, 90)) * 1000,
        "final_text_p50_ms": statistics.median(r["final_text_s"] for r in results) * 1000,
    }


# run 명령(T34 오프라인). 언어마다 클립을 고르고, 헛 미리 처리 쉼을 소리 1분당으로 센 뒤, 두 인식 설정과
# 번역을 한 번씩 미리 돌려 준비 시간을 뺀다. 그다음 (인식 설정 × 갱신 간격) 조합마다 모든 클립을 simulate해
# 요약 행과 클립별 상세를 모은다.
def run(args: argparse.Namespace) -> None:
    silero = Silero()
    model, translator = load_models()
    rows, skipped, pauses, details = [], {}, {}, {}
    for lang in args.langs:
        target = "ko" if lang == "en" else "en"
        began = time.perf_counter()
        chosen, skipped[lang] = utterances(silero, model, translator, lang, args.split, args.count)
        speech_min = sum(len(u["audio"]) for _, u, _ in chosen) / RATE / 60
        pauses[lang] = sum(u["pauses"] for _, u, _ in chosen) / speech_min
        # Warm up both settings on this language before any call is timed.
        for options in PARTIAL.values():
            recognize(model, chosen[0][1]["audio"], lang, options)
        translate(translator, chosen[0][2]["text"], lang, target)
        took = time.perf_counter() - began
        print(f"{lang}: {len(chosen)} clips, skipped {skipped[lang]}, {took:.0f} s", flush=True)
        for partial in args.partials:
            for interval in args.intervals:
                began = time.perf_counter()
                results = [
                    simulate(model, translator, u, final, lang, target, interval / 1000, PARTIAL[partial])
                    for _, u, final in chosen
                ]
                rows.append({"interval_ms": interval, "partial": partial, "lang": lang, **summarize(results)})
                details[f"{partial}_{interval}_{lang}"] = [
                    {"key": key, **r} for (key, _, _), r in zip(chosen, results, strict=True)
                ]
                took = time.perf_counter() - began
                print(f"  {partial} {interval}ms: {took:.0f} s", flush=True)
    write_report(rows, skipped, pauses, details, args)


# run 결과를 reports/stream_<tag>_<UTC 시각>.json·.md로 쓴다. Markdown에는 지표 정의를 함께 적어 표만 보고도
# 뜻을 알 수 있게 한다.
def write_report(
    rows: list[dict], skipped: dict, pauses: dict, details: dict, args: argparse.Namespace
) -> None:
    gpu, _ = gpu_memory_mb()
    stamp = datetime.now(UTC)
    base = REPORTS / f"stream_{args.tag}_{stamp:%Y%m%d_%H%M%S}"
    report = {"gpu": gpu, "skipped": skipped, "pauses_per_min": pauses, "rows": rows, "details": details}
    base.with_suffix(".json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    lines = [
        f"# 실시간 자막 오프라인 측정 ({args.tag})",
        "",
        f"- 날짜: {stamp.isoformat()}, FLEURS {args.split}, 언어별 {args.count}문장, GPU {gpu}",
        f"- 뺀 문장: {skipped}",
        "- 한 마디 안에서 헛 미리 처리를 부르는 쉼(192~992ms), 소리 1분당: "
        + ", ".join(f"{lang} {value:.1f}" for lang, value in pauses.items()),
        "- 모두 대소문자·띄어쓰기·문장부호를 뺀 글자로 센다. 표시 지연: 최종 단어의 글자가 화면에 모두 "
        "나타나 계속 남게 된 첫 시각 − 소리에서 그 단어가 끝난 시각. 흔들림: 지운 글자 ÷ 최종 글자. "
        "확정 뒤 바뀜: 진하게 보인 뒤 다음 글에서 (글자 정렬로) 찾을 수 없게 된 글자 ÷ 진하게 된 글자. "
        "모델 사용: 호출 시간 ÷ 소리 길이",
        "",
        "| 갱신 간격 | 인식 | 언어 | 문장 | 표시 지연 p50 | p90 | 원문 흔들림 | 원문 확정 뒤 바뀜 | "
        "번역 흔들림 | 번역 확정 뒤 바뀜 | 모델 사용 | 초당 갱신 | 최종 대기 p50/p90 | 최종 글자까지 p50 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['interval_ms']}ms | {r['partial']} | {r['lang']} | {r['utterances']} | "
            f"{r['lag_p50_ms']:.0f}ms | {r['lag_p90_ms']:.0f}ms | {r['source_erasure']:.2f} | "
            f"{r['source_dark_dropped']:.1%} | {r['translation_erasure']:.2f} | "
            f"{r['translation_dark_dropped']:.1%} | {r['model_use']:.2f} | {r['updates_per_s']:.1f} | "
            f"{r['final_wait_p50_ms']:.0f}/{r['final_wait_p90_ms']:.0f}ms | {r['final_text_p50_ms']:.0f}ms |"
        )
    base.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written to {base.with_suffix('.md')}")


# FRAME_S: 32ms 조각 길이(초). QUIET_END_FRAMES: 1000ms 쉼을 조각 수로 바꾼 31. 브라우저는 마지막 말소리 뒤
# 31조각 조용하면 말 끝으로 판정한다.
FRAME_S = S_CHUNK / RATE
QUIET_END_FRAMES = round(SILENCE_MS / (FRAME_S * 1000))  # 31: the browser decides the end here


# 실제 서비스 측정에서 브라우저(silero.ts)가 보낼 메시지 계획을 만든다. 출력: (그때까지 들은 조각 수, 종류,
# 보낼 소리 조각 범위) 목록. 종류는 utterance(마디 시작), audio, pause(미리 처리 요청), resume(다시 말함, 미리
# 처리 버림), end(1초 확정).
# 핵심은 '말 끝 뒤 6조각까지만 보내고 나머지 조용한 조각은 들고 있다가 말이 다시 시작될 때만 보낸다'는
# 규칙이다. 그래서 어느 쉼에서 pause를 보내도 서버가 가진 소리는 업로드되던 클립과 정확히
# 같다(tests/test_stream_eval.py가 확인). pause_frames는 T61 후보 Q(미리 처리를 시작하는 쉼 조각 수)다.
def browser_plan(flags, detected: int, pause_frames: int) -> list[tuple[int, str, tuple[int, int] | None]]:
    """What the browser sends for a clip, in order: (frames heard by then, kind, frames of audio).

    As frontend/src/conversation/silero.ts does: the second speech frame starts the utterance with the
    frames before it; quiet frames past the six after speech are held back and sent only if speech
    resumes, so at any pause the server has exactly the clip. The final is asked for after `pause_frames`
    of quiet (T61 candidate Q, at least six), counting on past the clip's end, and the end comes at 31.
    """
    plan: list[tuple[int, str, tuple[int, int] | None]] = []
    quiet = 0
    for i, flag in enumerate(flags):
        heard = i + 1
        # 말 시작 판정 전에는 보내지 않는다. 판정 순간에 마디 시작 메시지와 그때까지의 소리(앞 192ms 포함)를
        # 한꺼번에 보낸다.
        if heard < detected:
            continue
        if heard == detected:
            plan += [(heard, "utterance", None), (heard, "audio", (0, heard))]
            continue
        # 말소리 조각: 직전 쉼이 pause를 보낼 만큼 길었으면 resume을 먼저 보내고, 들고 있던 조용한 조각(6개를
        # 넘은 것)까지 이 조각과 함께 보낸다.
        if flag:
            if quiet >= pause_frames:
                plan.append((heard, "resume", None))
            plan.append((heard, "audio", (i - max(0, quiet - PAD_FRAMES), heard)))  # held frames too
            quiet = 0
            continue
        # 조용한 조각: 말 끝 뒤 6조각까지만 바로 보내고, 쉼이 pause_frames에 이르면 pause를 보낸다.
        quiet += 1
        if quiet <= PAD_FRAMES:
            plan.append((heard, "audio", (i, heard)))
        if quiet == pause_frames:
            plan.append((heard, "pause", None))
    # 클립이 끝난 뒤에도 브라우저는 조용함을 계속 듣는다. 클립 밖의 조용한 조각을 1초(31조각)까지 세며, 그
    # 사이에 pause 시점이 오면 보낸다. 마지막으로 마지막 말소리 뒤 31조각에서 end.
    for heard in range(len(flags) + 1, len(flags) - quiet + QUIET_END_FRAMES + 1):  # quiet past the clip
        quiet += 1
        if quiet == pause_frames:
            plan.append((heard, "pause", None))
    plan.append((len(flags) - PAD_FRAMES + QUIET_END_FRAMES, "end", None))
    return plan


# 클립 하나를 웹소켓으로 브라우저처럼 실제 시간에 맞춰 보내고, 클립 시작부터 서버 메시지가 도착한 시각을 잰다.
# 출력: 결과 종류, 단어별 표시 지연, 말 끝 → final, 말 끝 → 번역 음성 받기 완료, pause·resume 수, 자막 갱신
# 수, 서버 결과, 음성의 SHA-256.
# 호출 위치: stream_all(service), run_modes(modes의 미리 처리 경로).
async def stream_one(
    ws, http, auth: dict, number: int, utterance: dict, final: dict, pause_frames: int = PAD_FRAMES
) -> dict:
    """Send one clip as the browser would, a 32 ms frame at a time in real time, with its pause, resume and
    end messages, and time what comes back from the clip's start."""
    loop = asyncio.get_running_loop()
    pcm = np.round(utterance["audio"] * 32768).astype("<i2")
    flags = utterance["flags"]
    detected = round(utterance["detected_s"] / FRAME_S)
    messages: list[tuple[float, dict]] = []
    t0 = loop.time()

    # 받는 쪽: 모든 메시지를 도착 시각과 함께 쌓고, 이 마디 번호의 final 또는 error가 오면 끝낸다. 보내기와
    # 따로 돌려 자막 메시지가 보내는 도중에도 시각이 찍히게 한다.
    async def receive() -> dict:
        while True:
            message = json.loads(await ws.recv())
            messages.append((loop.time() - t0, message))
            if message.get("id") == number and message["type"] in ("final", "error"):
                return message

    # count번째 조각을 들은 실제 시각까지 기다린다(시작 시각 기준이라 오차가 쌓이지 않는다).
    async def at_frame(count: int) -> None:
        await asyncio.sleep(max(0.0, t0 + count * FRAME_S - loop.time()))

    receiver = asyncio.create_task(receive())
    pauses = resumes = 0
    for at, kind, frames in browser_plan(flags, detected, pause_frames):
        await at_frame(at)
        if kind == "audio":
            await ws.send(pcm[frames[0] * S_CHUNK : frames[1] * S_CHUNK].tobytes())
        else:
            await ws.send(json.dumps({"type": kind, "id": number}))
            pauses += kind == "pause"
            resumes += kind == "resume"  # each drops the final prepared at the pause before (T77)
    # 결과를 최대 120초 기다린다. 번역 음성이 있으면 인증된 요청으로 받아 받기 완료 시각을 재고, 해시를
    # 남긴다(경로 간 음성 바이트 비교는 참고용).
    outcome = await asyncio.wait_for(receiver, 120)
    final_at, audio_at, audio_sha256 = messages[-1][0], None, None
    if outcome["type"] == "final" and outcome["result"]["audio_id"]:
        audio = await http.get(f"/api/audio/{outcome['result']['audio_id']}", headers=auth)
        audio.raise_for_status()
        audio_at = loop.time() - t0
        audio_sha256 = hashlib.sha256(audio.content).hexdigest()
    speech_end = final["spans"][-1][2]
    lags = None
    # 서버의 최종 원문이 이 PC의 최종 인식과 글자가 같을 때만 표시 지연을 계산한다. 다르면 단어 위치를 맞출 수
    # 없어 보고서에 '원문 다름'으로 센다.
    if outcome["type"] == "final" and letters(outcome["result"]["source_text"]) == letters(final["text"]):
        events = [
            (at, letters(m["text"])) for at, m in messages if m["type"] == "source" and m["id"] == number
        ]
        lags = appearance_lags(events + [(final_at, letters(final["text"]))], final["spans"])
    return {
        "outcome": outcome["type"],
        "lags": lags,
        "final_s": final_at - speech_end,
        "audio_s": None if audio_at is None else audio_at - speech_end,
        "pauses": pauses,
        "updates": sum(1 for _, m in messages if m["type"] == "source"),
        "resumes": resumes,
        "result": outcome.get("result"),
        "detail": outcome.get("detail"),
        "audio_sha256": audio_sha256,
    }


# service 명령의 실제 전송. 측정용 계정을 매번 새로 만들어(임의의 이메일·비밀번호) 로그인하고, 언어마다 웹소켓
# 하나를 열어 start → ready 뒤 클립을 차례로 보낸다. 마디 사이 0.5초는 번역 음성 재생 같은 사이를 흉내 낸
# 것이다.
async def stream_all(args: argparse.Namespace, chosen: dict) -> dict:
    import httpx
    from websockets.asyncio.client import connect

    base = args.base_url.rstrip("/")
    results = {}
    async with httpx.AsyncClient(base_url=base, timeout=60) as http:
        email, password = f"live-{uuid.uuid4().hex[:8]}@example.com", secrets.token_urlsafe(16)
        (
            await http.post("/api/auth/register", json={"email": email, "password": password})
        ).raise_for_status()
        login = await http.post("/api/auth/login", data={"username": email, "password": password})
        token = login.json()["access_token"]
        auth = {"Authorization": f"Bearer {token}"}
        for lang, clips in chosen.items():
            target = "ko" if lang == "en" else "en"
            async with connect("ws" + base.removeprefix("http") + "/api/translate/live", max_size=None) as ws:
                start = {"type": "start", "token": token, "source_lang": lang, "target_lang": target}
                await ws.send(json.dumps(start))
                assert json.loads(await ws.recv())["type"] == "ready"
                results[lang] = []
                for number, (key, utterance, final) in enumerate(clips, 1):
                    result = await stream_one(ws, http, auth, number, utterance, final, args.pause_frames)
                    results[lang].append({"key": key, **result})
                    await asyncio.sleep(0.5)  # a moment between utterances, like playback would take
            print(f"{lang}: {len(results[lang])} clips streamed", flush=True)
    return results


# 측정 중 GPU 사용률·메모리와 CPU 사용률을 0.5초마다 기록하는 컨텍스트 관리자(T63). 같은 GPU를 다른
# 프로그램(게임 등)이 쓰던 때의 측정이 크게 흔들렸던 일(8절 T63, README 알려진 한계)이 있어, 지연 측정에
# 다른 프로그램의 간섭이 있었는지 함께 보려고 만들었다.
# GPU를 쓰는 프로세스 이름도 시작·끝에 모으지만, 그 이름은 공개 보고서가 아니라 커밋하지 않는 work/에만 쓴다.
class GpuLog:
    """GPU use while streaming, sampled twice a second (T63). The server's own work shows in it too: what
    tells of another program (a game on the same GPU) is use that stays high between utterances."""

    # NVML·psutil을 준비하고 샘플링 스레드를 띄운다. daemon 스레드라 측정이 예외로 끝나도 프로세스 종료를 막지
    # 않는다.
    def __enter__(self) -> GpuLog:
        import threading

        import psutil
        import pynvml

        self.nvml = pynvml
        self.psutil = psutil
        self.cpu: list[float] = []
        psutil.cpu_percent(None)  # the first call only starts the count
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.samples: list[tuple[float, float]] = []
        self.at_start = set(self.processes())
        self.programs: list[str] = []
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    # 멈춤 신호가 올 때까지 0.5초마다 GPU 사용률(%)·사용 메모리(MB)와 CPU 사용률을 적는다. Event.wait를 써서
    # 멈춤 신호에 바로 반응한다.
    def sample(self) -> None:
        while not self.stopped.wait(0.5):
            use = self.nvml.nvmlDeviceGetUtilizationRates(self.handle)
            memory = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
            self.samples.append((use.gpu, memory.used / 2**20))
            self.cpu.append(self.psutil.cpu_percent(None))  # the whole machine, since the last sample

    # 지금 GPU를 쓰는 그래픽·연산 프로세스의 이름 목록. 이름을 읽을 수 없는 프로세스는 pid로 남긴다.
    def processes(self) -> list[str]:
        import psutil

        names = []
        for query in ("nvmlDeviceGetGraphicsRunningProcesses", "nvmlDeviceGetComputeRunningProcesses"):
            for process in getattr(self.nvml, query)(self.handle):
                try:
                    names.append(psutil.Process(process.pid).name())
                except (psutil.Error, OSError):
                    names.append(f"pid {process.pid}")
        return names

    # 샘플링을 멈추고, 시작·끝에 본 GPU 프로세스 이름을 합친다.
    def __exit__(self, *exc) -> None:
        self.stopped.set()
        self.thread.join()
        self.programs = sorted(self.at_start | set(self.processes()))

    # 보고서에 실을 요약: 샘플 수, GPU 사용률 평균·p90·최대, 메모리 최대, CPU 평균. 샘플이 없으면 0으로 둔다.
    def summary(self) -> dict:
        use = [u for u, _ in self.samples] or [0.0]
        return {
            "samples": len(self.samples),
            "mean_util": float(np.mean(use)),
            "p90_util": float(np.percentile(use, 90)),
            "max_util": float(max(use)),
            "max_memory_mb": max((m for _, m in self.samples), default=0.0),
            "mean_cpu": float(np.mean(self.cpu)) if self.cpu else 0.0,
        }


# service가 흘려보낼 클립과 이 PC 모델로 구한 최종 결과(단어 시각)를 고른다. 모델은 이 함수 안의 지역 변수라
# 돌아오면 참조가 사라진다.
def choose_clips(silero: Silero, args: argparse.Namespace) -> tuple[dict, dict]:
    """The clips the service check streams, per language, with this machine's final results (word times)."""
    model, translator = load_models()
    chosen, skipped = {}, {}
    for lang in args.langs:
        chosen[lang], skipped[lang] = utterances(silero, model, translator, lang, args.split, args.count)
    return chosen, skipped


# service 명령. 클립을 고른 뒤 gc.collect로 이 프로세스의 모델 메모리를 놓고 나서 흘려보낸다. 예전에는 모델을
# 쥔 채 흘려보냈는데, Hy-MT2를 더한 서버와 함께면 GPU 11GB를 넘을 수 있어 T83에서 바꿨다(15절 계획 변경 ⑨).
# tests/test_stream_eval.py가 이 순서를 확인한다.
# 언어별로 오류 수, 원문 다름 수, 표시 지연 p50·p90, 말 끝 → final p50, 말 끝 → 음성 p50·p90, pause 수를
# 보고서에 쓴다.
def service(args: argparse.Namespace) -> None:
    """docs/experiments.md 8, real service check: the composed service through nginx, in real time.
    The clips and their word times come from this machine's models first. They are released before streaming
    (T83), so their GPU memory is not held next to the server's while it is measured."""
    chosen, skipped = choose_clips(Silero(), args)
    gc.collect()
    with GpuLog() as gpu:
        results = asyncio.run(stream_all(args, chosen))
    # Other programs' names stay out of the published report: work/ is not committed.
    (PROJECT / "work" / f"gpu_programs_{args.tag}.txt").write_text("\n".join(gpu.programs), encoding="utf-8")
    rows = []
    for lang, items in results.items():
        lags = [lag for item in items if item["lags"] for lag in item["lags"]]
        finals = [item["final_s"] for item in items if item["outcome"] == "final"]
        audio = [item["audio_s"] for item in items if item["audio_s"] is not None]
        rows.append(
            {
                "lang": lang,
                "utterances": len(items),
                "errors": sum(1 for item in items if item["outcome"] != "final"),
                "text_differs": sum(
                    1 for item in items if item["outcome"] == "final" and item["lags"] is None
                ),
                "lag_p50_ms": statistics.median(lags) * 1000,
                "lag_p90_ms": float(np.percentile(lags, 90)) * 1000,
                "final_p50_ms": statistics.median(finals) * 1000,
                "audio_p50_ms": statistics.median(audio) * 1000,
                "audio_p90_ms": float(np.percentile(audio, 90)) * 1000,
                "pauses": sum(item["pauses"] for item in items),
            }
        )
    stamp = datetime.now(UTC)
    base = REPORTS / f"stream_{args.tag}_{stamp:%Y%m%d_%H%M%S}"
    report = {"skipped": skipped, "gpu_use": gpu.summary(), "rows": rows, "details": results}
    base.with_suffix(".json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    lines = [
        f"# 동시통역 실제 서비스 측정 ({args.tag})",
        "",
        f"- 날짜: {stamp.isoformat()}, FLEURS {args.split}, 언어별 {args.count}문장, {args.base_url}",
        f"- 미리 처리를 부르는 쉼: {args.pause_frames}조각({args.pause_frames * 32}ms). "
        "서버 설정은 따로 적는다",
        "- GPU (측정 중 0.5초마다, 이 서버의 사용 포함): 사용률 평균 {mean_util:.0f}%, p90 {p90_util:.0f}%, "
        "최대 {max_util:.0f}%, 메모리 최대 {max_memory_mb:.0f}MB".format(**gpu.summary()),
        f"- 뺀 문장: {skipped}",
        "- 표시 지연: 오프라인과 같은 정의(서버 최종 원문이 이 PC의 최종 인식과 글자가 다른 마디는 뺌). "
        "말 끝 → 최종·음성: 마지막 단어 끝부터 final 메시지, 번역 음성 받기 완료까지",
        "",
        "| 언어 | 문장 | 오류 | 원문 다름 | 표시 지연 p50 | p90 | "
        "말 끝→final p50 | 말 끝→음성 p50 | p90 | pause 수 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['lang']} | {r['utterances']} | {r['errors']} | {r['text_differs']} | "
            f"{r['lag_p50_ms']:.0f}ms | {r['lag_p90_ms']:.0f}ms | {r['final_p50_ms']:.0f}ms | "
            f"{r['audio_p50_ms']:.0f}ms | {r['audio_p90_ms']:.0f}ms | {r['pauses']} |"
        )
    base.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written to {base.with_suffix('.md')}")


# 여기부터 modes 명령(T77)에 쓰는 상수·함수다.
# T77 (docs/experiments.md 11): conversation and dialog modes, each clip once over the old path (the browser
# uploads the WAV over HTTP after the 1000 ms end) and once over the new one (streamed over the live
# WebSocket without subtitles, prepared at the 192 ms pause, committed at the end), back to back.

# MODES: 잴 화면 모드. OTHER: 원문 언어 → 번역 언어.
# SAVING_MS: 채택 규칙의 '두 방향 모두 p50이 300ms 이상 빨라짐'. UNDECIDABLE_LIMIT: 방향마다 허용하는 '판단
# 불가' 클립 수(2개). SECOND_USER_EVERY_S: 간섭 측정에서 두 번째 사용자가 요청을 보내는 간격(11절 지표).
MODES = ("conversation", "dialog")
OTHER = {"ko": "en", "en": "ko"}
SAVING_MS = 300  # the rule: both directions' p50 at least this much sooner
UNDECIDABLE_LIMIT = 2  # clips per direction whose HTTP result itself changes from call to call
SECOND_USER_EVERY_S = 5.0
# What a clip's result must share over both paths. The synthesized audio itself cannot be compared:
# MeloTTS samples noise (and durations) at every call, so the same text gives different WAV bytes.
SAME_FIELDS = (
    "source_lang",
    "target_lang",
    "source_text",
    "translated_text",
    "stt_model",
    "mt_model",
    "tts_model",
    "tts_error",
    "language_guessed",
)


# 옛 경로에서 브라우저가 올리는 WAV를 만든다. 서버가 웹소켓으로 받은 PCM으로 만드는 파일과 같은
# 함수(wav_from_pcm)라 두 경로가 정확히 같은 소리를 받는다.
def clip_wav(audio: np.ndarray) -> bytes:
    """The WAV the browser uploads for a clip (frontend pcm.ts encodeWav): the same file the server builds
    from the streamed PCM (app/services/live.py)."""
    return wav_from_pcm(np.round(audio * 32768).astype("<i2").tobytes())


# 브라우저가 1초 쉼으로 말 끝을 판정하는 조각 수(browser_plan의 마지막 end 시점). 옛 경로는 이때 업로드를
# 시작한다.
def end_frame(utterance: dict) -> int:
    """Frames heard when the browser decides the end (1000 ms of quiet), as browser_plan sends it."""
    detected = round(utterance["detected_s"] / FRAME_S)
    return browser_plan(utterance["flags"], detected, PAD_FRAMES)[-1][0]


# 두 경로가 같아야 하는 부분만 뽑는다: 실패면 결과 종류와 실패 문구, 성공이면 음성 유무와 SAME_FIELDS(원문,
# 번역문, 언어, 모델 이름, 합성 오류, 언어 추정 여부). 합성 음성 바이트는 MeloTTS가 호출마다 잡음을 뽑아
# 달라지므로 넣지 않는다(11절 계획 변경).
def comparable(outcome: dict) -> dict:
    """What must be the same over both paths: the error, or the result's texts, languages and models."""
    if outcome["outcome"] != "final":
        return {"outcome": outcome["outcome"], "detail": outcome.get("detail")}
    result = outcome["result"]
    return {
        "outcome": "final",
        "has_audio": result.get("audio_id") is not None,
        **{key: result.get(key) for key in SAME_FIELDS},
    }


# 두 결과가 비교할 부분에서 같은지.
def same(a: dict, b: dict) -> bool:
    return comparable(a) == comparable(b)


# 클립 하나의 동일성 분류. 두 경로가 같으면 'same'. 다르면 같은 WAV를 HTTP로 두 번 더 보낸 결과(rechecks)를
# 본다: 두 번 모두 처음 업로드와 같으면 미리 처리가 결과를 바꾼 것이라 'different', 아니면 HTTP 경로 자체가
# 같은 소리에 다른 답을 내는 모델 비결정성이라 'undecidable'(11절 지표).
def classify(upload: dict, prepared: dict, rechecks: list[dict]) -> str:
    """'same', or for a clip whose paths differ: 'different' when two more uploads of its WAV give the
    first upload's result again (the prepared path changed it), else 'undecidable' (the HTTP path itself
    gives other results for the same WAV: model nondeterminism, not the prepare)."""
    if same(upload, prepared):
        return "same"
    if len(rechecks) == 2 and all(same(upload, again) for again in rechecks):
        return "different"
    return "undecidable"


# 클립별 차이(업로드 − 미리 처리)의 중앙값과 그 95% 백분위 붓스트랩 구간(10,000번, 시드 77, 11절 지표).
# 판정에는 쓰지 않는 참고 값이다.
def paired_ci(differences: list[float], seed: int = 77, rounds: int = 10_000) -> tuple[float, float, float]:
    """The median of per-clip differences and its 95% percentile bootstrap interval."""
    values = np.asarray(differences, dtype=float)
    draws = np.random.default_rng(seed).choice(values, size=(rounds, len(values)), replace=True)
    medians = np.median(draws, axis=1)
    return float(np.median(values)), float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


# 백분위를 ms로. 한 경로가 음성을 하나도 내지 못했으면 NaN을 돌려주고, NaN은 decide의 비교에서 늘 거짓이라
# 규칙을 통과하지 못한다.
def quantile_ms(values: list[float], q: float) -> float:
    """A percentile in ms, or NaN when a path produced no audio at all (it then fails the rule)."""
    return float(np.percentile(values, q)) * 1000 if values else float("nan")


# 모드 × 원문 언어마다 한 행으로 요약한다: 두 경로의 말 끝 → 음성 p50·p90과 그 차이(saving_ms), 짝지은 차이
# 중앙값과 구간, 말 끝 → final p50, 동일성 분류 수, 음성 바이트가 같은 수(참고), 경로별 오류 수, 미리 처리
# 수(pause)와 버린 수(resume).
def summarize_modes(rows: list[dict]) -> list[dict]:
    """One row per mode and source language."""
    out = []
    for mode in MODES:
        for lang in LANGS:
            items = [r for r in rows if r["mode"] == mode and r["lang"] == lang]
            if not items:
                continue
            upload = [r["upload"]["audio_s"] for r in items if r["upload"]["audio_s"] is not None]
            prepared = [r["prepared"]["audio_s"] for r in items if r["prepared"]["audio_s"] is not None]
            # 두 경로 모두 음성을 받은 클립만 짝지어 차이를 낸다.
            pairs = [
                r["upload"]["audio_s"] - r["prepared"]["audio_s"]
                for r in items
                if r["upload"]["audio_s"] is not None and r["prepared"]["audio_s"] is not None
            ]
            median, low, high = paired_ci(pairs) if pairs else (float("nan"),) * 3
            classes = [r["identity"] for r in items]
            out.append(
                {
                    "mode": mode,
                    "lang": lang,
                    "clips": len(items),
                    "upload_p50_ms": quantile_ms(upload, 50),
                    "upload_p90_ms": quantile_ms(upload, 90),
                    "prepared_p50_ms": quantile_ms(prepared, 50),
                    "prepared_p90_ms": quantile_ms(prepared, 90),
                    "saving_ms": quantile_ms(upload, 50) - quantile_ms(prepared, 50),
                    "paired_median_ms": median * 1000,
                    "paired_ci_ms": (low * 1000, high * 1000),
                    "upload_final_p50_ms": quantile_ms([r["upload"]["final_s"] for r in items], 50),
                    "prepared_final_p50_ms": quantile_ms([r["prepared"]["final_s"] for r in items], 50),
                    "same": classes.count("same"),
                    "different": classes.count("different"),
                    "undecidable": classes.count("undecidable"),
                    "audio_bytes_same": sum(
                        1
                        for r in items
                        if r["upload"].get("audio_sha256")
                        and r["upload"].get("audio_sha256") == r["prepared"].get("audio_sha256")
                    ),
                    "errors_upload": sum(1 for r in items if r["upload"]["outcome"] != "final"),
                    "errors_prepared": sum(1 for r in items if r["prepared"]["outcome"] != "final"),
                    "prepares": sum(r["prepared"]["pauses"] for r in items),
                    "dropped": sum(r["prepared"]["resumes"] for r in items),
                }
            )
    return out


# 11절 규칙 중 이 도구가 재는 부분(시간·동일성)으로 모드마다 채택 여부와 떨어진 이유를 돌려준다. 세 번째
# 조건인 e2e 회귀는 eval/e2e_eval.py 보고서로 따로 판정한다.
def decide(rows: list[dict]) -> dict[str, dict]:
    """The rule written before measuring (docs/experiments.md 11), for the parts this tool measures: per
    mode, adopt when in both directions the upload path's end-to-audio p50 is at least 300 ms later than
    the prepared path's, no clip is 'different' and at most two are 'undecidable'. The e2e_eval regression
    check is the rule's third part and is judged from its own report."""
    verdict = {}
    for mode in MODES:
        mine = [r for r in rows if r["mode"] == mode]
        if not mine:
            continue
        reasons = []
        if {r["lang"] for r in mine} != set(LANGS):
            reasons.append("not both directions")
        for r in mine:
            # 'not (x >= 300)'으로 쓴 것은 saving_ms가 NaN일 때도 조건 불충족으로 잡히게 하려는 것이다('x <
            # 300'은 NaN에서 거짓이 된다).
            if not r["saving_ms"] >= SAVING_MS:
                reasons.append(f"{r['lang']}: p50 {r['saving_ms']:.0f}ms sooner, under {SAVING_MS}ms")
            if r["different"]:
                reasons.append(f"{r['lang']}: {r['different']} clips differ")
            if r["undecidable"] > UNDECIDABLE_LIMIT:
                reasons.append(f"{r['lang']}: {r['undecidable']} undecidable clips, over {UNDECIDABLE_LIMIT}")
        verdict[mode] = {"adopt": not reasons, "reasons": reasons}
    return verdict


# 옛 경로(업로드) 한 클립: 브라우저가 1초 쉼으로 말 끝을 판정할 때까지 아무것도 보내지 않다가, WAV를 HTTP
# API로 올리고 번역 음성을 받는다. 시각은 stream_one과 같이 클립 시작 기준이고, 결과에는 말 끝 기준으로 바꿔
# 넣는다.
# 대화 모드는 /api/translate/speech에 방향을 주고, 두 사람 대화는 /api/translate/dialog에 직전 마디
# 언어(previous_lang)를 준다(서버가 언어 판별이 애매할 때 쓰는 값).
async def upload_one(http, auth: dict, mode: str, lang: str, utterance: dict, final: dict, previous) -> dict:
    """The old path for one clip: nothing is sent until the browser decides the end, then the WAV goes to
    the HTTP API and the translated speech is fetched. Times from the clip's start, as stream_one's."""
    loop = asyncio.get_running_loop()
    wav = clip_wav(utterance["audio"])
    t0 = loop.time()
    await asyncio.sleep(max(0.0, t0 + end_frame(utterance) * FRAME_S - loop.time()))
    sent_at = loop.time() - t0
    files = {"audio": ("conversation.wav", wav, "audio/wav")}
    if mode == "conversation":
        path, data = "/api/translate/speech", {"source_lang": lang, "target_lang": OTHER[lang]}
    else:
        path, data = "/api/translate/dialog", ({"previous_lang": previous} if previous else {})
    reply = await http.post(path, headers=auth, files=files, data=data)
    final_at = loop.time() - t0
    speech_end = final["spans"][-1][2]
    out: dict = {"sent_s": sent_at - speech_end, "final_s": final_at - speech_end, "audio_s": None}
    # 201이 아니면 오류로 기록한다. 실패 문구는 동일성 비교에 쓰므로 JSON이면 detail을 남긴다.
    if reply.status_code != 201:
        try:
            detail = reply.json().get("detail")
        except ValueError:
            detail = None
        return out | {"outcome": "error", "status": reply.status_code, "detail": detail}
    result = reply.json()
    out |= {"outcome": "final", "result": result}
    if result["audio_id"]:
        audio = await http.get(f"/api/audio/{result['audio_id']}", headers=auth)
        audio.raise_for_status()
        out["audio_s"] = loop.time() - t0 - speech_end
        out["audio_sha256"] = hashlib.sha256(audio.content).hexdigest()
    return out


# 두 경로 결과가 다른 클립의 같은 WAV를 HTTP로 한 번 더 보낸다(기다림 없이). classify의 재확인에 쓴다.
async def reupload(http, auth: dict, mode: str, lang: str, utterance: dict, previous) -> dict:
    """The same clip over HTTP once more, at once, for the identity check of a clip whose paths differ."""
    files = {"audio": ("conversation.wav", clip_wav(utterance["audio"]), "audio/wav")}
    if mode == "conversation":
        path, data = "/api/translate/speech", {"source_lang": lang, "target_lang": OTHER[lang]}
    else:
        path, data = "/api/translate/dialog", ({"previous_lang": previous} if previous else {})
    reply = await http.post(path, headers=auth, files=files, data=data)
    if reply.status_code != 201:
        return {"outcome": "error", "detail": reply.json().get("detail")}
    return {"outcome": "final", "result": reply.json()}


# 측정용 일회용 계정을 만들고 (접근 토큰, 인증 헤더)를 돌려준다. 이메일 앞부분(prefix)으로 어떤 측정의
# 계정인지 구분한다.
async def account(http, prefix: str) -> tuple[str, dict]:
    email, password = f"{prefix}-{uuid.uuid4().hex[:8]}@example.com", secrets.token_urlsafe(16)
    (await http.post("/api/auth/register", json={"email": email, "password": password})).raise_for_status()
    login = await http.post("/api/auth/login", data={"username": email, "password": password})
    token = login.raise_for_status().json()["access_token"]
    return token, {"Authorization": f"Bearer {token}"}


# 측정 내내 0.2초마다 GET /api/health를 보내 응답 시간을 그때의 구간 이름(phase)과 함께 남긴다. 모델 작업이
# 다른 요청을 막지 않는지 보는 T23 확인이다. 실패나 200이 아닌 응답은 무한대 시간으로 센다. done 신호를
# wait_for로 기다려 멈춤에 바로 반응한다.
async def watch_health(http, phase: list[str], out: list[tuple[str, float]], done: asyncio.Event) -> None:
    """GET /api/health every 0.2 s (T23's check that model work does not hold other requests)."""
    while not done.is_set():
        label, began = phase[0], time.perf_counter()
        try:
            reply = await http.get("/api/health")
            took = time.perf_counter() - began if reply.status_code == 200 else float("inf")
        except Exception:  # noqa: BLE001 - a failed check counts as an endless one
            took = float("inf")
        out.append((label, took))
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(done.wait(), 0.2)


# 같은 서버의 다른 사용자를 흉내 낸다: e2e 측정용 8~12초 음성(data/e2e/<split>)을 5초마다 HTTP 음성 번역 API로
# 보내고, 그때의 구간 이름과 응답 시간을 남긴다(11절 'T23 동시성 간섭', 참고용). 응답이 5초보다 길면 기다리지
# 않고 바로 다음 요청을 보낸다.
async def second_user(http, auth: dict, split: str, phase: list[str], out: list[dict], done: asyncio.Event):
    """Another person on the same server: an 8-12 s recording to the HTTP speech API every 5 s."""
    folder = DATA / "e2e" / split
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    loop = asyncio.get_running_loop()
    number = 0
    while not done.is_set():
        item = manifest[number % len(manifest)]
        number += 1
        label, began = phase[0], loop.time()
        files = {"audio": (item["file"], (folder / item["file"]).read_bytes(), "audio/wav")}
        data = {"source_lang": item["language"], "target_lang": OTHER[item["language"]]}
        reply = await http.post("/api/translate/speech", headers=auth, files=files, data=data)
        took = loop.time() - began
        body = reply.json() if reply.status_code == 201 else {}
        out.append({"phase": label, "elapsed_s": took, "status": reply.status_code, "id": body.get("id")})
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(done.wait(), max(0.0, SECOND_USER_EVERY_S - took))


# 웹소켓 연결별 클립 목록을 만든다. 대화 모드는 방향마다 연결 하나, 두 사람 대화는 연결 하나에 두 언어 클립을
# 번갈아 넣는다(zip_longest로 한쪽이 짧아도 남은 클립을 뒤에 둔다). 두 사람이 번갈아 말하는 상황이라 직전 마디
# 언어가 매번 바뀐다.
def turns(mode: str, chosen: dict) -> list[tuple[str, list]]:
    """Connections and their clips: conversation mode one per direction, dialog one with the two languages
    taking turns, as two people would."""
    if mode == "conversation":
        return [(lang, [(key, lang, u, f) for key, u, f in chosen[lang]]) for lang in chosen]
    lists = [[(key, lang, u, f) for key, u, f in chosen[lang]] for lang in chosen]
    mixed = [item for group in itertools.zip_longest(*lists) for item in group if item is not None]
    return [("dialog", mixed)]


# modes 명령의 실제 측정. 계정을 만들고 health 감시(와 선택 시 두 번째 사용자)를 띄운 뒤, 모드·연결마다 클립을
# 하나씩 두 경로로 잰다. 끝나면 결과가 다른 클립만 다시 보내 동일성을 분류하고, 측정이 만든 번역 기록을
# 지운다.
async def run_modes(args: argparse.Namespace, chosen: dict) -> tuple[list[dict], dict]:
    import httpx
    from websockets.asyncio.client import connect

    base = args.base_url.rstrip("/")
    # phase: 지금 재는 구간 이름을 감시 작업들과 나누는 공유 칸(목록 한 칸이라 다른 코루틴에서 바뀐 값을 읽을
    # 수 있다). 'idle'은 경로를 재지 않는 사이다.
    phase = ["idle"]
    health: list[tuple[str, float]] = []
    others: list[dict] = []
    rows: list[dict] = []
    async with httpx.AsyncClient(base_url=base, timeout=120) as http:
        token, auth = await account(http, "modes")
        other = auth  # the second user's account, when there is one
        done = asyncio.Event()
        watchers = [asyncio.create_task(watch_health(http, phase, health, done))]
        if args.second_user:
            _, other = await account(http, "second")
            watchers.append(asyncio.create_task(second_user(http, other, args.split, phase, others, done)))
        for mode in args.modes:
            for name, items in turns(mode, chosen):
                start = {"type": "start", "token": token, "mode": mode}
                if mode == "conversation":
                    start |= {"source_lang": name, "target_lang": OTHER[name]}
                async with connect(
                    "ws" + base.removeprefix("http") + "/api/translate/live", max_size=None
                ) as ws:
                    await ws.send(json.dumps(start))
                    assert json.loads(await ws.recv())["type"] == "ready"
                    previous = None  # dialog over HTTP: the browser's last answered language
                    for number, (key, lang, utterance, final) in enumerate(items, 1):
                        row = {"mode": mode, "lang": lang, "key": key, "number": number, "previous": previous}
                        # Alternate which path goes first, so a slow stretch of time hits both alike.
                        for path in ("upload", "prepared") if number % 2 else ("prepared", "upload"):
                            phase[0] = f"{mode}:{path}"
                            if path == "upload":
                                row[path] = await upload_one(
                                    http, auth, mode, lang, utterance, final, previous
                                )
                            else:
                                row[path] = await stream_one(ws, http, auth, number, utterance, final)
                            phase[0] = "idle"
                            await asyncio.sleep(0.5)  # a moment between utterances, like playback would take
                        # 두 사람 대화의 HTTP 경로는 브라우저처럼 직전 응답의 원문 언어를 다음 요청의
                        # previous_lang으로 넘긴다.
                        if mode == "dialog" and row["upload"]["outcome"] == "final":
                            previous = row["upload"]["result"]["source_lang"]
                        rows.append(row)
                        print(f"  {mode} {lang} {number}/{len(items)}", flush=True)  # a hang shows here
                print(f"{mode} {name}: {len(items)} clips", flush=True)
        done.set()
        await asyncio.gather(*watchers)
        # Clips whose paths differ: the same WAV over HTTP twice more (the identity check's rule).
        clips = {(lang, key): utterance for lang in chosen for key, utterance, _ in chosen[lang]}
        for row in rows:
            row["rechecks"] = []
            if not same(row["upload"], row["prepared"]):
                for _ in range(2):
                    clip = clips[(row["lang"], row["key"])]
                    row["rechecks"].append(
                        await reupload(http, auth, row["mode"], row["lang"], clip, row["previous"])
                    )
            row["identity"] = classify(row["upload"], row["prepared"], row["rechecks"])
        # The two throwaway accounts' records go; the accounts stay in the database (as T58's do).
        ids = [
            o["result"]["id"]
            for r in rows
            for o in (r["upload"], r["prepared"], *r["rechecks"])
            if o.get("result")
        ]
        for record in ids:
            await http.delete(f"/api/history/{record}", headers=auth)
        for item in others:
            if item["id"]:
                await http.delete(f"/api/history/{item['id']}", headers=other)
    return rows, {"health": health, "second_user": others}


# health 응답 시간을 구간별(idle 제외)로 수·p95·최대로 모은다.
def phase_stats(samples: list[tuple[str, float]]) -> dict[str, dict]:
    stats = {}
    for label in sorted({label for label, _ in samples} - {"idle"}):
        values = [value for name, value in samples if name == label]
        stats[label] = {
            "count": len(values),
            "p95_s": float(np.percentile(values, 95)),
            "max_s": float(max(values)),
        }
    return stats


# modes 명령. 이 PC 모델로 클립과 단어 시각을 만든 뒤 모델을 지워 GPU 메모리를 놓고(VRAM 경합을 줄임, 두
# 경로가 같은 조건이라 비교에는 영향 없음, 11절 계획 변경), GPU 기록을 켠 채 실제 측정을 돌린다. GPU 프로그램
# 이름은 work/에만 쓴다.
def modes(args: argparse.Namespace) -> None:
    """docs/experiments.md 11: the composed service through nginx, in real time, both paths per clip."""
    silero = Silero()
    model, translator = load_models()
    chosen, skipped = {}, {}
    for lang in args.langs:
        chosen[lang], skipped[lang] = utterances(silero, model, translator, lang, args.split, args.count)
    # The clips and their word times are ready: free this process's GPU memory before the service runs.
    del model, translator
    gc.collect()
    with GpuLog() as gpu:
        rows, watched = asyncio.run(run_modes(args, chosen))
    (PROJECT / "work" / f"gpu_programs_{args.tag}.txt").write_text("\n".join(gpu.programs), encoding="utf-8")
    write_modes_report(args, skipped, gpu.summary(), rows, watched, REPORTS)


# modes 결과를 요약·판정해 reports/stream_<tag>_*.json·.md로 쓰고 .md 경로를 돌려준다. 표 두 개: 모드·언어별
# 시간·동일성 표, 구간별 health와 두 번째 사용자 응답 표. 끝에 규칙의 시간·동일성 부분 판정을 적는다. 쓸
# 폴더를 인자로 받아 테스트가 임시 폴더로 부를 수 있다.
def write_modes_report(
    args: argparse.Namespace, skipped: dict, gpu_use: dict, rows: list[dict], watched: dict, folder: Path
) -> Path:
    table = summarize_modes(rows)
    verdict = decide(table)
    health = phase_stats(watched["health"])
    # 두 번째 사용자 응답 시간을 구간별로 모은다(성공한 201 응답만).
    others = {}
    for label in sorted({o["phase"] for o in watched["second_user"]} - {"idle"}):
        values = [
            o["elapsed_s"] for o in watched["second_user"] if o["phase"] == label and o["status"] == 201
        ]
        if values:
            others[label] = {
                "count": len(values),
                "p50_s": statistics.median(values),
                "p90_s": float(np.percentile(values, 90)),
            }
    stamp = datetime.now(UTC)
    base = folder / f"stream_{args.tag}_{stamp:%Y%m%d_%H%M%S}"
    report = {
        "split": args.split,
        "skipped": skipped,
        "gpu_use": gpu_use,
        "rows": table,
        "verdict": verdict,
        "health": health,
        "second_user": others,
        "details": rows,
        "watched": watched,
    }
    base.with_suffix(".json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    lines = [
        f"# 대화 모드·두 사람 대화 미리 처리 측정 ({args.tag})",
        "",
        f"- 날짜: {stamp.isoformat()}, FLEURS {args.split}, 언어별 {args.count}문장, {args.base_url}, "
        f"두 번째 사용자 {'있음' if args.second_user else '없음'}",
        "- 경로: 업로드 = 1초 판정 뒤 HTTP로 WAV(지금까지의 화면), "
        "미리 처리 = 웹소켓으로 흘리고 192ms 쉼에서 미리 처리, 1초에서 확정. 문장마다 두 경로를 번갈아 먼저",
        "- GPU (0.5초마다, 서버 포함): 사용률 평균 {mean_util:.0f}%, p90 {p90_util:.0f}%, "
        "최대 {max_util:.0f}%, 메모리 최대 {max_memory_mb:.0f}MB, CPU 평균 {mean_cpu:.0f}%".format(**gpu_use),
        f"- 뺀 문장: {skipped}",
        "- 말 끝 → 음성: 마지막 단어 끝부터 번역 음성 받기 완료까지. "
        "차이 중앙값은 문장별 (업로드 − 미리 처리)의 중앙값과 95% 붓스트랩 구간",
        "- 같음: 원문·번역문·언어·모델·합성 오류·추정 여부·음성 유무가 같음. "
        "다르면 같은 WAV를 HTTP로 두 번 더 보내 업로드 결과가 되풀이되면 '다름', "
        "아니면 '판단 불가'(모델 비결정성)",
        "",
        "| 모드 | 원문 | 문장 | 업로드 p50 | p90 | 미리 처리 p50 | p90 | p50 차이 | "
        "차이 중앙값 (95% 구간) | 같음/다름/판단 불가 | 음성 바이트 같음 | 오류 (업/미) | "
        "미리 처리 / 버림 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def ms(value: float) -> str:
        return f"{value:.0f}ms"

    for r in table:
        low, high = r["paired_ci_ms"]
        cells = [
            r["mode"],
            r["lang"],
            r["clips"],
            ms(r["upload_p50_ms"]),
            ms(r["upload_p90_ms"]),
            ms(r["prepared_p50_ms"]),
            ms(r["prepared_p90_ms"]),
            ms(r["saving_ms"]),
            f"{ms(r['paired_median_ms'])} ({low:.0f}~{high:.0f})",
            f"{r['same']}/{r['different']}/{r['undecidable']}",
            r["audio_bytes_same"],
            f"{r['errors_upload']}/{r['errors_prepared']}",
            f"{r['prepares']} / {r['dropped']}",
        ]
        lines.append("| " + " | ".join(str(cell) for cell in cells) + " |")
    lines += [
        "",
        "| 구간 | health 수 | p95 | 최대 | 두 번째 사용자 p50 | p90 | 수 |",
        "|---|---|---|---|---|---|---|",
    ]
    nan = float("nan")
    for label in sorted(set(health) | set(others)):
        h, o = health.get(label, {}), others.get(label, {})
        cells = [
            label,
            h.get("count", 0),
            f"{h.get('p95_s', nan):.3f}s",
            f"{h.get('max_s', nan):.3f}s",
            f"{o.get('p50_s', nan):.2f}s",
            f"{o.get('p90_s', nan):.2f}s",
            o.get("count", 0),
        ]
        lines.append("| " + " | ".join(str(cell) for cell in cells) + " |")
    lines += ["", "규칙의 시간·동일성 부분 (e2e 회귀는 e2e_eval 보고서로 따로 본다):"]
    for mode, v in verdict.items():
        lines.append(
            f"- {mode}: " + ("채택 조건 충족" if v["adopt"] else "불충족: " + "; ".join(v["reasons"]))
        )
    base.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written to {base.with_suffix('.md')}")
    return base.with_suffix(".md")


# 명령줄 진입점. 세 하위 명령 공통: --split, --langs, --tag(필수), --count(언어별 문장 수). run은
# --intervals·--partials로 조합을, service는 --base-url과 --pause-frames(T61 후보 Q)를, modes는
# --base-url·--modes·--second-user를 받는다. 기본 주소 8080은 compose의 nginx(web) 포트다.
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "service", "modes"):
        command = sub.add_parser(name)
        command.add_argument("--split", choices=("validation", "test"), default="validation")
        command.add_argument("--langs", nargs="+", choices=tuple(LANGS), default=list(LANGS))
        command.add_argument("--tag", required=True)
        command.add_argument("--count", type=int, default=PER_LANGUAGE)
    # The test split gets only the chosen combination (docs/experiments.md 8).
    sub.choices["run"].add_argument("--intervals", nargs="+", type=int, default=list(INTERVALS_MS))
    sub.choices["run"].add_argument("--partials", nargs="+", choices=tuple(PARTIAL), default=list(PARTIAL))
    sub.choices["service"].add_argument("--base-url", default="http://localhost:8080")
    # T61 candidate Q: quiet frames before the browser asks for the final (6 = 192 ms, the clip's end).
    sub.choices["service"].add_argument("--pause-frames", type=int, default=PAD_FRAMES)
    sub.choices["modes"].add_argument("--base-url", default="http://localhost:8080")
    sub.choices["modes"].add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    # T23's interference check: another person's HTTP requests during both paths (report only).
    sub.choices["modes"].add_argument("--second-user", action="store_true")
    args = parser.parse_args()
    {"run": run, "service": service, "modes": modes}[args.command](args)


if __name__ == "__main__":
    main()

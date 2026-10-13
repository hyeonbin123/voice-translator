"""What the live service check sends for a clip, as the browser would (T61, T62). No models, no GPU."""

# [파일 역할] 실제 서비스 측정 도구 eval/stream_eval.py가 브라우저처럼 보내는 메시지 계획과, 대화 모드·두 사람
# 대화의 '192ms 쉼에서 미리 처리' 채택 규칙(docs/experiments.md 11절)을 계산하는 함수들을 검증한다. 모델도
# GPU도 쓰지 않는다.
# [왜 중요한가] 측정 도구가 브라우저(frontend의 silero.ts)와 다른 소리를 보내면 서버 측정이 실제 사용과
# 달라지고, 판정 함수가 틀리면 기록된 채택 결정이 달라진다. 특히 '어느 쉼에서 미리 처리를 요청해도 서버가 가진
# 소리는 업로드되던 클립과 정확히 같다'는 성질이 미리 처리 결과가 업로드 결과와 같을 수 있는 전제다.

import numpy as np
import pytest

# wav_from_pcm: 서버가 웹소켓으로 받은 PCM으로 WAV를 만드는 실제 함수. RATE(16 kHz)·S_CHUNK(32 ms 조각)는
# 브라우저 Silero 판정과 같은 상수다. PAD_FRAMES는 말 앞뒤로 붙이는 6조각(192 ms), QUIET_END_FRAMES는 말 끝
# 판정에 필요한 1초 쉼(31조각)이다.
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
# 조각별 말소리 판정: 조용 6, 말 10, 안쪽 쉼 8, 말 5, 조용 6(클립 끝의 여유분). 말은 조각 6부터지만 연속
# 2조각을 들어야 시작으로 판정하므로 브라우저는 조각 8(DETECTED)에서 보내기 시작한다.
FLAGS = np.array([False] * 6 + [True] * 10 + [False] * 8 + [True] * 5 + [False] * 6)
DETECTED = 8
INNER_QUIET = 8


# 미리 처리를 요청하는 쉼 길이(6·12·19조각)마다 확인한다: 마지막 메시지는 마지막 말소리 뒤 31조각의 end이고,
# 마지막 pause는 그보다 앞서 마지막 말소리 뒤 pause_frames 조각에 온다(클립이 끝난 뒤에도 쉼을 계속 센다).
# 안쪽 쉼(8조각)이 pause_frames 이상일 때만 안쪽 pause와 다시 말할 때의 resume이 한 번씩 더 생긴다. 계획은
# 판정 시점(조각 8)의 utterance·audio로 시작한다.
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


# 마지막 pause 시점까지 보낸 audio 조각을 이어 붙이면 클립의 모든 조각이 순서대로 한 번씩이고(대화 모드가
# 업로드할 WAV와 같음), 그 뒤로는 소리를 더 보내지 않는지 확인한다. 말 끝 뒤 6조각을 넘는 조용한 조각은 들고
# 있다가 말이 다시 시작될 때만 보내는 규칙 덕분이다.
@pytest.mark.parametrize("pause_frames", [6, 12, 19])
def test_at_the_last_pause_the_server_has_exactly_the_clip(pause_frames):
    plan = browser_plan(FLAGS, DETECTED, pause_frames)
    last_pause = max(i for i, (_, kind, _) in enumerate(plan) if kind == "pause")
    sent = [frame for _, kind, span in plan[:last_pause] if kind == "audio" for frame in range(*span)]
    # Every frame of the clip once, in order: the WAV conversation mode would upload.
    assert sent == list(range(len(FLAGS)))
    assert not any(kind == "audio" for _, kind, _ in plan[last_pause:])


# T77 (docs/experiments.md 11): the two paths per clip and the rule, without a server.


# 고정 시드의 무작위 파형(16비트 PCM 정밀도로 양자화)과 FLAGS로 클립 하나와, 192 ms 쉼(PAD_FRAMES)에서 미리
# 처리하는 브라우저 계획을 만든다.
def clip_and_plan():
    rng = np.random.default_rng(7)
    audio = quantize(rng.uniform(-0.5, 0.5, len(FLAGS) * S_CHUNK).astype(np.float32))
    utterance = {"audio": audio, "flags": FLAGS, "detected_s": DETECTED * S_CHUNK / RATE}
    return utterance, browser_plan(FLAGS, DETECTED, PAD_FRAMES)


# 옛 경로(1초 판정 뒤 HTTP 업로드)의 WAV와 새 경로(웹소켓으로 흘려 마지막 pause에서 서버가 만드는 WAV)가
# 바이트까지 같은지 확인한다. 두 경로가 같은 소리를 받아야 결과 비교가 공정하다. 업로드 시점(end_frame)도 새
# 경로의 end 메시지 시점과 같아야 두 경로의 시간 비교 기준이 같다.
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


# 서버의 성공 응답(final) 모양을 만드는 도우미. audio=False면 합성이 실패한 응답, guessed를 주면 두 사람
# 대화의 언어 추측 여부가 붙는다.
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


# 두 경로 결과의 동일성 비교 대상: 원문·번역문·언어·모델 이름·합성 오류·언어 추측 여부와 음성 유무. 기록 id,
# 처리 시간, 음성 바이트는 호출마다 달라지므로 보지 않는다(MeloTTS는 호출마다 잡음을 뽑아 같은 글자도 WAV가
# 다르다, stream_eval의 SAME_FIELDS 주석). 같은 실패 문구의 오류끼리는 같고, 음성 유무나 추측 여부가 다르면
# 다르다.
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


# 두 경로가 다를 때의 분류: 같은 WAV를 HTTP로 두 번 더 보내 두 번 모두 처음 업로드와 같으면 미리 처리가 결과를
# 바꾼 것('different'), 다시 보낸 결과가 흔들리거나 확인이 없으면 업로드 경로 자체가 같은 소리에 다른 답을
# 내는 모델 비결정성이라 'undecidable'이다. 미리 처리 탓으로 돌리는 것은 증거가 있을 때만이다.
def test_a_differing_clip_counts_against_the_prepare_only_when_uploads_repeat_the_first_answer():
    upload, prepared = answer("하나"), answer("둘")
    assert classify(upload, prepared, [answer("하나"), answer("하나")]) == "different"
    assert classify(upload, prepared, [answer("하나"), answer("둘")]) == "undecidable"
    assert classify(upload, prepared, [answer("셋"), answer("셋")]) == "undecidable"
    assert classify(upload, prepared, []) == "undecidable"


# 클립별 시간 차이의 중앙값과 그 붓스트랩 구간: 중앙값이 맞고(0.71), 구간 안에 있으며, 시드 고정이라 다시
# 계산해도 같다. 판정에는 쓰지 않는 참고 값이다.
def test_the_paired_interval_is_seeded_and_holds_the_median():
    differences = [0.6, 0.7, 0.8, 0.65, 0.9, 0.75, 0.7, 0.72]
    median, low, high = paired_ci(differences)
    assert median == pytest.approx(0.71)
    assert low <= median <= high
    assert paired_ci(differences) == (median, low, high)


# 측정 결과 한 행(클립 하나)을 만드는 도우미. 업로드 경로와 미리 처리 경로의 '말 끝 → 번역 음성'
# 시간(audio_s), 미리 처리 횟수(pauses)와 버린 횟수(resumes)를 정한다.
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


# 모드 × 언어마다 count개의 클립 행을 만들고 summarize_modes로 요약한다. 언어별로 미리 처리가 줄인
# 시간(saving)과 첫 행의 동일성 분류를 바꿀 수 있다.
def table(saving_en=0.7, saving_ko=0.7, identity="same", count=3, modes=("conversation",)):
    rows = []
    for mode in modes:
        for lang, saving in (("en", saving_en), ("ko", saving_ko)):
            rows += [row(mode, lang, 2.0 + i / 10, 2.0 + i / 10 - saving, resumes=i) for i in range(count)]
    if identity != "same":
        rows[0]["identity"] = identity
    return summarize_modes(rows)


# 요약 행 확인: 업로드 p50 2.1초와 미리 처리 p50 1.4초의 차이 700 ms(짝지은 중앙값도 700), 미리 처리 횟수 합
# 6(1+2+3)과 버린 횟수 합 3(0+1+2), 동일성 분류 수.
def test_the_summary_counts_prepares_dropped_prepares_and_the_saving():
    [en, ko] = table()
    assert (en["mode"], en["lang"], ko["lang"]) == ("conversation", "en", "ko")
    assert en["saving_ms"] == pytest.approx(700) and en["paired_median_ms"] == pytest.approx(700)
    assert (en["upload_p50_ms"], en["prepared_p50_ms"]) == (pytest.approx(2100), pytest.approx(1400))
    assert (en["prepares"], en["dropped"]) == (6, 3)
    assert (en["same"], en["different"], en["undecidable"]) == (3, 0, 0)


# 측정 전에 정한 채택 규칙(11절) 중 이 도구가 판정하는 부분: 두 방향 모두 p50이 300 ms 이상 빨라지고(정확히
# 300은 통과, 299는 실패), 'different' 클립이 하나도 없고, 'undecidable'은 방향마다 2개까지. 한 방향만 잰
# 모드는 채택하지 않고, 모드마다 따로 판정한다. 세 번째 조건(e2e 회귀)은 다른 보고서로 판정한다.
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


# 웹소켓 연결별 클립 순서: 두 사람 대화는 연결 하나에 영어·한국어 클립을 번갈아 넣고(남은 클립은 뒤에), 대화
# 모드는 방향마다 연결을 따로 둔다. 두 사람이 번갈아 말하는 실제 상황처럼 직전 마디 언어가 매번 바뀌게 하려는
# 것이다.
def test_dialog_clips_take_turns_between_the_languages_on_one_connection():
    chosen = {"en": [("e1", "u", "f"), ("e2", "u", "f")], "ko": [("k1", "u", "f")]}
    [(name, items)] = turns("dialog", chosen)
    assert name == "dialog" and [(key, lang) for key, lang, _, _ in items] == [
        ("e1", "en"),
        ("k1", "ko"),
        ("e2", "en"),
    ]
    assert [name for name, _ in turns("conversation", chosen)] == ["en", "ko"]


# 회귀 테스트(커밋 6a0f541): 실제 서비스 측정(service)은 이 PC의 모델로 클립과 단어 시각을 고른 뒤, 서버로
# 흘려보내기 전에 그 모델을 놓아야 한다. 잡고 있으면 측정 중인 서버 옆에서 GPU 메모리를 차지한다.
# 방법: 모델 적재·클립 고르기·GPU 기록을 가짜로 바꾸고, stream_all이 불리는 순간 gc.collect() 뒤 두 모델의
# 약한 참조가 모두 비었는지 본다. 확인한 뒤 Stop 예외로 실제 흘려보내기와 보고서 쓰기를 건너뛴다. 마지막
# 단언은 모델이 실제로 두 개 만들어졌는지(가짜 적재가 쓰였는지)를 확인한다.
def test_the_service_check_releases_its_own_models_before_streaming(monkeypatch):
    # T83: this machine's models only pick the clips and their word times; held while streaming, they took
    # GPU memory next to the server's (the Hy-MT2 trial adds a model to the server's Ollama).
    import argparse
    import gc
    import weakref

    from eval import stream_eval

    class Model:
        pass

    made: list[weakref.ref] = []

    def load_models():
        model, translator = Model(), Model()
        made.extend([weakref.ref(model), weakref.ref(translator)])
        return model, translator

    class Stop(Exception):
        pass

    def stream_all(args, chosen):
        gc.collect()
        assert [ref() for ref in made] == [None, None]
        raise Stop

    class NoGpuLog:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(stream_eval, "Silero", lambda: None)
    monkeypatch.setattr(stream_eval, "load_models", load_models)
    monkeypatch.setattr(stream_eval, "utterances", lambda *args: ([], {"not_one_utterance": 0}))
    monkeypatch.setattr(stream_eval, "stream_all", stream_all)
    monkeypatch.setattr(stream_eval, "GpuLog", NoGpuLog)
    args = argparse.Namespace(langs=["en", "ko"], split="validation", count=40)
    with pytest.raises(Stop):
        stream_eval.service(args)
    assert len(made) == 2

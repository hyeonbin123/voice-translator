"""실제 서비스 확인이 브라우저처럼 클립 하나에 대해 보내는 것을 검증한다(T61, T62). 모델·GPU 없이 돈다."""

# [파일 역할] 실제 서비스 측정 도구 eval/stream_eval.py가 브라우저처럼 보내는 메시지 계획과, 대화 모드·두 사람
# 대화의 '192ms 쉼에서 미리 처리' 채택 규칙(docs/experiments.md 11절)을 계산하는 함수들을 검증한다. 모델도
# GPU도 쓰지 않는다.
# [왜 중요한가] 측정 도구가 브라우저(frontend의 silero.ts)와 다른 소리를 보내면 서버 측정이 실제 사용과
# 달라지고, 판정 함수가 틀리면 기록된 채택 결정이 달라진다. 특히 '어느 쉼에서 미리 처리를 요청해도 서버가 가진
# 소리는 업로드되던 클립과 정확히 같다'는 성질이 미리 처리 결과가 업로드 결과와 같을 수 있는 전제다.

import json

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
    inner = INNER_QUIET >= pause_frames
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
    assert sent == list(range(len(FLAGS)))
    assert not any(kind == "audio" for _, kind, _ in plan[last_pause:])


# 여기부터 T77(docs/experiments.md 11절): 클립마다의 두 경로와 판단 규칙을 서버 없이 검증한다.


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
    assert decide(table()[:1])["conversation"]["adopt"] is False  # 한 방향만 있음
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
# 흘려보내기 전에 그 모델을 놓아야 한다. 잡고 있으면 측정 중인 서버 옆에서 GPU 메모리를 차지한다(T83: Hy-MT2
# 시험은 서버의 Ollama에 모델을 하나 더 올린다).
# 방법: 모델 적재·클립 고르기·GPU 기록을 가짜로 바꾸고, stream_all이 불리는 순간 gc.collect() 뒤 두 모델의
# 약한 참조가 모두 비었는지 본다. 확인한 뒤 Stop 예외로 실제 흘려보내기와 보고서 쓰기를 건너뛴다. 마지막
# 단언은 모델이 실제로 두 개 만들어졌는지(가짜 적재가 쓰였는지)를 확인한다.
def test_the_service_check_releases_its_own_models_before_streaming(monkeypatch):
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


# 여기부터 17절(말 끝 → 번역 음성 단계별 진단)에 더한 것.


# service의 첫 메시지: live는 mode를 넣지 않아 8절 이후의 측정과 같은 메시지이고, conversation은 mode를 넣어
# 자막 없는 대화 모드로 연다. 번역 방향은 원문의 반대 언어다.
def test_the_service_start_message_names_the_mode_only_for_conversation():
    from eval.stream_eval import start_message

    assert start_message("t", "en", "live") == {
        "type": "start",
        "token": "t",
        "source_lang": "en",
        "target_lang": "ko",
    }
    assert start_message("t", "ko", "conversation") == {
        "type": "start",
        "token": "t",
        "source_lang": "ko",
        "target_lang": "en",
        "mode": "conversation",
    }


# 가짜 웹소켓: 보낸 메시지를 모으고, end를 받으면 그 마디의 final(서버 timings 포함)을 돌려준다.
class FakeSocket:
    def __init__(self, timings):
        import asyncio

        self.sent: list = []
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.timings = timings

    async def send(self, message):
        self.sent.append(message)
        if isinstance(message, str) and '"end"' in message:
            result = {"source_text": "hello", "audio_id": None}
            await self.inbox.put(
                json.dumps({"type": "final", "id": 1, "result": result, "timings": self.timings})
            )

    async def recv(self):
        return await self.inbox.get()


# 클립 하나를 흘려보내면, 진단에 쓰는 마지막 pause와 end를 보낸 시각(말 끝 기준)이 계획대로 기록되고, 서버가
# final에 실어 보낸 timings가 그대로 결과에 남는다. 시간을 아끼려고 조각 길이(FRAME_S)를 1ms로 줄여 돌린다.
async def test_streaming_records_when_the_pause_and_end_were_sent_and_keeps_the_server_timings(monkeypatch):
    from eval import stream_eval

    monkeypatch.setattr(stream_eval, "FRAME_S", 0.001)
    utterance, plan = clip_and_plan()
    utterance["detected_s"] = DETECTED * 0.001
    speech_end_s = 0.020
    final = {"text": "hello", "spans": [(0, 5, speech_end_s)]}
    timings = {"from": "pause", "stages": {}}
    ws = FakeSocket(timings)
    result = await stream_eval.stream_one(ws, None, {}, 1, utterance, final)
    assert result["outcome"] == "final" and result["timings"] == timings
    last_pause_frame = max(at for at, kind, _ in plan if kind == "pause")
    end_frame_ = plan[-1][0]
    assert result["pause_sent_s"] == pytest.approx(last_pause_frame * 0.001 - speech_end_s, abs=0.05)
    assert result["end_sent_s"] == pytest.approx(end_frame_ * 0.001 - speech_end_s, abs=0.05)
    assert result["end_sent_s"] > result["pause_sent_s"]


# 여기부터 18절(g2pkk 정규식 캐시 재측정)에 더한 것: GPU 기록기(GpuLog)가 사용률·메모리 옆에 SM·메모리 클럭,
# P-state, 클럭 제한 이유를 함께 남기는지 가짜 pynvml로 본다. 같은 소리의 서버 인식 시간이 동시통역과 대화
# 모드에서 수십 ms 달랐는데(17절), 기록에 GPU 클럭 상태가 없어 모드 효과와 GPU 상태를 가를 수 없었다.
# CI에는 nvidia-ml-py가 없으므로 sys.modules의 pynvml 자리에 가짜 모듈을 넣는다.


def fake_pynvml(*, util, sm=None, memory_clock=None, pstate=None, event=None, throttle=None):
    """값 목록을 차례로 돌려주는 가짜 pynvml 모듈.

    sm·memory_clock·pstate·event(새 이름 nvmlDeviceGetCurrentClocksEventReasons)·throttle(옛 이름
    nvmlDeviceGetCurrentClocksThrottleReasons)이 None이면 그 함수가 없는 판(옛 pynvml)이다. 목록의 "error"는
    그 호출에서 NVMLError를 낸다(드라이버에 없는 함수, 읽기 실패). 클럭·P-state·제한 이유 목록의 첫 값은
    GpuLog가 시작할 때 한 번 읽어 보는 데 쓰인다.
    """
    import types

    class NVMLError(Exception):
        pass

    def reader(values):
        values = iter(values)

        def read(*args):
            value = next(values)
            if value == "error":
                raise NVMLError("not supported")
            return value

        return read

    module = types.ModuleType("pynvml")
    module.NVMLError = NVMLError
    module.NVML_CLOCK_SM, module.NVML_CLOCK_MEM = 1, 2
    module.nvmlInit = lambda: None
    module.nvmlDeviceGetHandleByIndex = lambda index: "gpu0"
    module.nvmlDeviceGetGraphicsRunningProcesses = lambda handle: []
    module.nvmlDeviceGetComputeRunningProcesses = lambda handle: []
    utilization = iter(util)
    module.nvmlDeviceGetUtilizationRates = lambda handle: types.SimpleNamespace(gpu=next(utilization))
    module.nvmlDeviceGetMemoryInfo = lambda handle: types.SimpleNamespace(used=1024 * 2**20)
    if sm is not None:
        clocks = {1: reader(sm), 2: reader(memory_clock)}
        module.nvmlDeviceGetClockInfo = lambda handle, kind: clocks[kind]()
    if pstate is not None:
        module.nvmlDeviceGetPerformanceState = reader(pstate)
    if event is not None:
        module.nvmlDeviceGetCurrentClocksEventReasons = reader(event)
    if throttle is not None:
        module.nvmlDeviceGetCurrentClocksThrottleReasons = reader(throttle)
    return module


# 가짜 pynvml로 GpuLog를 열고 샘플을 reads번 직접 읽은 뒤 요약을 돌려준다. 샘플 간격을 한 시간으로 늘려
# 기록 스레드가 스스로 읽지 않게 하므로(닫을 때 바로 멈춤) 샘플 수가 늘 reads와 같다.
def gpu_log_summary(monkeypatch, nvml, reads: int) -> dict:
    import sys

    from eval import stream_eval

    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    monkeypatch.setattr(stream_eval.GpuLog, "INTERVAL_S", 3600.0)
    with stream_eval.GpuLog() as log:
        for _ in range(reads):
            log.read()
    return log.summary()


# 새 판 pynvml: SM 클럭 중앙값(모든 샘플과 사용률이 0보다 큰 샘플만), 메모리 클럭 중앙값, P-state별 샘플 수,
# 제한 이유가 있던 샘플의 비율(유휴 표시 gpu_idle만 있는 샘플은 제한으로 세지 않음)과 이유별 샘플 수가
# 요약에 들어가고, 원래 있던 사용률·메모리 요약은 그대로다. 보고서의 사람이 읽는 줄도 이 값을 쓴다.
def test_the_gpu_log_keeps_clocks_pstates_and_limit_reasons(monkeypatch):
    from eval.stream_eval import clock_line

    nvml = fake_pynvml(
        util=[0, 50, 80, 0],
        sm=[999, 210, 1900, 1950, 300],
        memory_clock=[999, 405, 7001, 7001, 405],
        pstate=[9, 8, 2, 0, 8],
        event=[0, 0x1, 0x0, 0x4, 0x1],
    )
    summary = gpu_log_summary(monkeypatch, nvml, reads=4)
    assert summary["samples"] == 4 and summary["mean_util"] == 32.5 and summary["max_memory_mb"] == 1024
    assert summary["sm_clock_p50_mhz"] == 1100.0
    assert summary["busy_sm_clock_p50_mhz"] == 1925.0
    assert summary["memory_clock_p50_mhz"] == pytest.approx((405 + 7001) / 2)
    assert summary["pstates"] == {"P0": 1, "P2": 1, "P8": 2}
    assert summary["limited_ratio"] == 0.25
    assert summary["clock_reasons"] == {"gpu_idle": 2, "sw_power_cap": 1}
    assert summary["reasons_api"] == "event" and summary["clock_read_errors"] == 0
    line = clock_line(summary)
    assert "1100MHz" in line and "1925MHz" in line and "25%" in line and "P8 2" in line


# 제한 이유는 새 이름 함수가 없는 판(옛 pynvml)이나, 함수는 있지만 드라이버에 없어 처음 읽을 때 NVMLError가
# 나는 경우(옛 드라이버) 옛 이름 함수로 읽는다.
@pytest.mark.parametrize("event", [None, ["error"]], ids=["old-pynvml", "old-driver"])
def test_the_gpu_log_falls_back_to_the_old_throttle_reason_name(monkeypatch, event):
    nvml = fake_pynvml(
        util=[10, 20],
        sm=[1, 1500, 1500],
        memory_clock=[1, 7000, 7000],
        pstate=[0, 2, 2],
        event=event,
        throttle=[0, 0x8, 0x0],
    )
    summary = gpu_log_summary(monkeypatch, nvml, reads=2)
    assert summary["reasons_api"] == "throttle"
    assert summary["clock_reasons"] == {"hw_slowdown": 1} and summary["limited_ratio"] == 0.5


# 클럭·P-state·제한 이유 함수가 하나도 없는 판이면 그 칸은 비우고(None, 빈 dict) 사용률 기록은 그대로 한다.
# 처음에는 읽혔는데 측정 중 한 번 읽지 못하면 그 샘플의 그 값만 비우고 실패 수를 센다(기록은 계속된다).
# 샘플이 없으면 새 칸도 비어 있다.
def test_the_gpu_log_leaves_out_what_nvml_cannot_read(monkeypatch):
    bare = gpu_log_summary(monkeypatch, fake_pynvml(util=[30, 40]), reads=2)
    assert bare["samples"] == 2 and bare["mean_util"] == 35.0
    assert bare["sm_clock_p50_mhz"] is None and bare["memory_clock_p50_mhz"] is None
    assert bare["pstates"] == {} and bare["clock_reasons"] == {} and bare["limited_ratio"] is None
    assert bare["reasons_api"] is None and bare["clock_read_errors"] == 0
    flaky = fake_pynvml(
        util=[30, 40, 50],
        sm=[1, 1500, "error", 1700],
        memory_clock=[1, 7000, 7000, 7000],
        pstate=[0, 2, 2, 2],
        event=[0, 0, 0, 0],
    )
    summary = gpu_log_summary(monkeypatch, flaky, reads=3)
    assert summary["sm_clock_p50_mhz"] == 1600.0 and summary["clock_read_errors"] == 1
    assert summary["pstates"] == {"P2": 3} and summary["limited_ratio"] == 0.0
    empty = gpu_log_summary(monkeypatch, fake_pynvml(util=[]), reads=0)
    assert empty["samples"] == 0 and empty["sm_clock_p50_mhz"] is None and empty["pstates"] == {}

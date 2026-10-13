"""The functions behind the recorded end-of-speech and live subtitle numbers (T33, T34, T61), on hand-made
cases. They need only numpy, so they run in CI without the eval group."""

# [파일 역할] 기록된 측정 수치를 만든 지표 함수를 손으로 만든 작은 사례로 고정한다: 대화 모드 말 끝
# 판정(eval/eos_eval.py의 group·score)과 동시통역 자막(eval/stream_eval.py의
# appearance_lags·dark_changes·erased).
# [왜 필요한가] README와 docs/experiments.md의 잘림 비율·판정 지연·자막 흔들림 수치가 모두 이 함수들에서
# 나온다. 함수가 바뀌면 같은 녹음으로 다른 숫자가 나오므로, 정답을 손으로 계산할 수 있는 입력으로 동작을 못
# 박는다.
# [설계] 두 모듈은 numpy만으로 지표를 계산하게 정리돼 있어(커밋 b431bd5: pyarrow import를 load_clips 안으로
# 옮김) eval 의존성 그룹이 없는 CI에서도 돈다.

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# S_CHUNK(512표본 = 32 ms 프레임)와 S_START_FRAMES(말 시작으로 볼 연속 말소리 프레임 수 2)는 브라우저 Silero
# 판정과 같은 값을 쓰려고 상수 그대로 가져온다.
from eval.eos_eval import S_CHUNK, S_START_FRAMES, group, score
from eval.stream_eval import appearance_lags, dark_changes, erased

# backend/ 폴더. 아래 자식 프로세스가 `eval` 패키지를 찾도록 작업 폴더로 쓴다.
BACKEND = Path(__file__).resolve().parents[1]


# pyarrow가 없는 환경(CI)에서도 두 지표 모듈을 import할 수 있는지 확인한다. 모듈 맨 위에서 pyarrow를
# import하는 회귀가 생기면 CI에서 이 파일 전체가 깨지기 때문이다.
# 새 파이썬 프로세스에서 sys.modules['pyarrow'] = None으로 pyarrow import를 강제로 실패시킨 뒤 두 모듈을
# 불러온다. 현재 테스트 프로세스에는 이미 불러온 모듈이 남아 있을 수 있어 별도 프로세스로 확인한다.
# CUDA_VISIBLE_DEVICES=-1은 import만 보는 자식 프로세스에 GPU가 보이지 않게 한다.
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


# 브라우저 Silero 판정과 같은 조건(32 ms 프레임, 말 끝으로 볼 쉼 1000 ms, 시작 2프레임)으로 group을 돌리고,
# 표본 단위 결과를 프레임 단위 (시작, 끝, 판정 시점)으로 바꿔 사람이 손으로 셀 수 있게 하는 도우미.
def silero_group(flags: list[bool]) -> list[tuple[int, int, int]]:
    """group() with the Silero frames (32 ms) and the chosen 1000 ms of quiet, in frames."""
    return [
        (u["start"] // S_CHUNK, u["end"] // S_CHUNK, u["decided"] // S_CHUNK)
        for u in group(np.array(flags), S_CHUNK, 1000, S_START_FRAMES)
    ]


# 기본 동작: 프레임 10~29가 말소리면 마디는 10에서 시작해 30(첫 조용한 프레임)에서 끝나고, 1000 ms =
# round(1000/32) = 31프레임 쉰 뒤인 61에서 끝이 판정된다. 판정 지연을 '말의 끝'부터 재는 기준이 맞는지
# 고정한다.
def test_an_utterance_ends_after_the_quiet_and_is_timed_from_its_first_speech_frame():
    # 1000 ms is 31 frames of 32 ms: the end is decided on the 31st quiet frame.
    assert silero_group([False] * 10 + [True] * 20 + [False] * 40) == [(10, 30, 61)]


# 말소리가 250 ms(MIN_SPEECH_MS) 미만이면 마디로 치지 않고 버린다(브라우저 endpointer.ts와 같은 규칙). 7프레임
# = 224 ms라 버려진다.
def test_speech_shorter_than_250_ms_is_dropped():
    assert silero_group([False] * 10 + [True] * 7 + [False] * 40) == []  # 7 frames = 224 ms


# 29초 상한(MAX_CLIP_MS, 앞에 붙이는 200 ms 여유분 6프레임 포함): 프레임 40부터 이어진 긴 말은 여유분을 더해
# 906프레임이 되는 940에서 쉼을 기다리지 않고 바로 잘라 판정하고(끝 = 판정 시점), 이어지는 말은 잘린
# 지점(940)부터 새 마디가 된다. 업로드하는 마디가 서버의 30초 제한을 넘지 않게 하는 분할이다(endpointer.ts
# 주석).
def test_the_29_second_limit_splits_long_speech():
    utterances = silero_group([False] * 40 + [True] * 2000 + [False] * 40)
    # 906 frames with the 6 pre-roll frames: the split is decided at once, and speech goes on in a new clip.
    assert utterances[0] == (40, 940, 940)
    assert utterances[1][0] == 940


# score의 각 수치를 한 번에 확인한다(시각은 16 kHz 표본 단위). 정답 문장 5개에 대해: 둘로 잘린 문장(split) 1,
# 놓친 문장(missed) 1, 두 문장이 한 마디로 붙은 이웃 쌍(merged_pairs) 1(분모 pairs는 이웃 쌍 수 4), 말이 없던
# 곳의 마디(false) 1.
# 판정 지연은 정답 문장 하나와 마디 하나가 1:1로 맞은 경우만 잰다. 첫 문장은 끝(1000) 뒤 16000표본 = 1초에
# 판정됐으므로 [1000.0] ms다. 붙은 마디(세 번째·네 번째)는 지연에서 빠진다.
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


# 동시통역 자막의 '진하게 보인(확정된 것처럼 보인) 글자가 나중에 바뀐 수'를 세는 규칙을 고정한다. 앞쪽 단어
# 하나가 바뀌어도 그 뒤의 진한 글자 전부가 아니라 실제로 바뀐 글자(p→v 1개)만 센다(docs/experiments.md 8절, 첫
# validation 뒤 고친 규칙). 결과 (1, 1)은 (바뀐 진한 글자, 새로 진해진 글자)다.
# erased는 그와 달리 화면 끝에서 지워진 글자 수를 그대로 센다: thejapanesepeople → thejavanesepeople에서 공통
# 앞부분 'theja'(5글자) 뒤 12글자가 지워졌다.
def test_a_changed_dark_word_counts_its_changed_letters_not_the_rest():
    # docs/experiments.md 8: "Japanese" heard again as "Javanese" changes one dark letter, not every dark
    # letter after it; erasure still counts what was taken off the end of the screen.
    shown = [("thejapanesepeople", 17), ("thejavanesepeoplelive", 17)]
    final = "thejavanesepeoplelive"
    assert dark_changes(shown, final) == (1, 1)
    assert erased(["thejapanese", "thejapanesepeople", "thejavanesepeople", final]) == 12


# 다음 갱신이 없어도 최종 결과가 진한 글자를 바꾸면 센다: 진한 "mould"가 최종 "moldis"에서 u를 잃었으므로 바뀐
# 글자 1, 새로 진해진 글자 0.
def test_dark_letters_the_final_result_changes_are_counted():
    assert dark_changes([("mouldis", 5)], "moldis") == (1, 0)


# 단어가 자막에 '나타난 시각'은 처음 보인 때가 아니라 그 뒤로 사라지지 않고 계속 보이기 시작한 때다. hello는
# 0.5초에 보였다가 1.0초에 사라져 1.5초가 기준이고(1.5 - 0.3 = 1.2), world는 최종 결과(2.0초)에서야 온전히
# 보였다(2.0 - 0.9 = 1.1). 깜빡인 자막을 빠르다고 세지 않게 하는 규칙이다.
def test_a_word_appears_when_it_is_on_screen_to_stay():
    events = [(0.5, "hello"), (1.0, "help"), (1.5, "hellowor"), (2.0, "helloworld")]
    spans = [(0, 5, 0.3), (5, 10, 0.9)]
    # "hello" went away at 1.0 and came back at 1.5; "world" was shown whole only by the final result.
    assert appearance_lags(events, spans) == pytest.approx([1.2, 1.1])


# 대화 모드 서버 응답 측정(eos_eval server)에서 한 조건(언어 × 길이)의 요청이 모두 실패해도 멈추지 않고, 그
# 조건의 중앙값을 None으로 둔 채 앞서 잰 조건과 함께 보고서를 쓰는지 확인한다. 예전에는 빈 목록의
# 중앙값(statistics.median)에서 StatisticsError로 멈춰 보고서가 남지 않았다. 서버는 httpx MockTransport로,
# 녹음과 말 경계는 가짜로 바꾼다. 영어 한 녹음으로 3초 조각(첫 번역 요청)은 201, 5초 조각(둘째)은 422다.
def test_the_server_report_survives_a_condition_whose_requests_all_failed(tmp_path, monkeypatch):
    import argparse
    import json

    import httpx

    from eval import eos_eval

    translations = []

    def answer(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/auth/register":
            return httpx.Response(201, json={})
        if request.url.path == "/api/auth/login":
            return httpx.Response(200, json={"access_token": "token"})
        translations.append(request.url.path)
        return httpx.Response(201 if len(translations) == 1 else 422, json={})

    client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: client(transport=httpx.MockTransport(answer), **kw))
    clip = np.zeros(6 * eos_eval.RATE, dtype=np.float32)
    monkeypatch.setattr(eos_eval, "load_clips", lambda lang, split: [("0-1", clip)])
    monkeypatch.setattr(
        eos_eval, "word_bounds", lambda clips, lang, split: {"0-1": [eos_eval.RATE, clip.size]}
    )
    monkeypatch.setattr(eos_eval, "REPORTS", tmp_path)
    args = argparse.Namespace(base_url="http://test", langs=["en"], split="validation", count=1, tag="t")
    eos_eval.server(args)
    results = json.loads(next(tmp_path.glob("eos_t_*.json")).read_text(encoding="utf-8"))["results"]
    assert translations == ["/api/translate/speech"] * 2
    assert results["en_3s"]["count"] == 1 and results["en_3s"]["p50_s"] >= 0
    assert results["en_5s"] == {"count": 0, "failures": 1, "p50_s": None}

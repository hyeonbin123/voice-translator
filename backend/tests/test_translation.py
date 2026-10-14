# [파일 역할] 번역 서비스 app/services/translation.py를 모델 파일 없이 검증한다. CTranslate2
# 엔진·SentencePiece는 가짜로, Ollama HTTP는 httpx.MockTransport로 바꾼다.
# [무엇을 확인하나] opus-mt(Marian)·NLLB 어댑터가 엔진 실패와 빈 번역을 ModelError로 바꾸는지, opus 학습 때의
# 전처리를 그대로 옮겼는지, 영→한 문장 단위 번역의 문장 나누기(약어·숫자에서 나누지 않음, docs/experiments.md
# 2-1절), 방향별 연결, 비교용 Ollama LLM 번역기와 시험용 Hy-MT2 번역기(15절)의 요청 형식·실패 처리·시작
# 준비(prepare) 시간 제한.
# [왜 중요한가] ModelError로 통일돼야 API가 500이 아니라 503으로 답하고 기록을 남기지 않는다. 요청
# 형식(프롬프트·옵션)은 측정한 설정과 같아야 측정 결과가 서비스에 그대로 들어맞는다.

import json
import time
from pathlib import Path

import httpx
import pytest

# translation 모듈은 monkeypatch 대상(_Ct2Model, load_sentencepiece)으로, 나머지는 시험할 클래스와 함수로
# 가져온다. FakeTranslator는 방향별 연결 시험에 쓴다.
from app.services import translation
from app.services.interfaces import ModelError
from app.services.translation import (
    DirectionalTranslator,
    HyMtTranslator,
    OllamaTranslator,
    marian_source_tokens,
    nllb_source_tokens,
    opus_preprocess,
    split_sentences,
)
from tests.fakes import FakeTranslator


# SentencePiece 대역: 공백으로 나눈 단어를 조각으로 쓰고, 공백으로 다시 잇는다.
class SplitProcessor:
    """SentencePiece 대신 쓰인다: 단어 하나가 조각 하나."""

    def encode(self, text, out_type):
        assert out_type is str
        return text.split()

    def decode(self, pieces):
        return " ".join(pieces)


# CTranslate2 대역: 정해 준 조각을 돌려주거나 정해 준 예외를 던진다.
class FakeEngine:
    """CTranslate2 모델 대신 쓰인다: 정해 둔 조각을 돌려주거나 예외를 던진다."""

    def __init__(self, result=None, error=None):
        self.result, self.error = result, error

    def generate(self, tokens, target_prefix=None):
        if self.error:
            raise self.error
        return self.result


# 엔진에 들어온 토큰을 기록하고, 끝 토큰을 뺀 조각을 대문자로 돌려주는 대역. 문장 단위 번역이 엔진을 몇 번,
# 어떤 입력으로 부르는지 확인할 때 쓴다.
class EchoEngine:
    """원문 조각을 대문자로 바꿔 번역으로 돌려주고, 호출마다 기록한다."""

    def __init__(self):
        self.calls = []

    def generate(self, tokens, target_prefix=None):
        self.calls.append(tokens)
        return [piece.upper() for piece in tokens[:-1]]  # 문장 끝 토큰은 뺀다


# 실제 MarianTranslator·NllbTranslator 클래스를 모델 파일 없이 만들 수 있게, 엔진 생성과 SentencePiece 적재를
# 가짜로 바꾼다. 돌려주는 함수로 테스트마다 쓸 가짜 엔진을 정한다.
@pytest.fixture
def engine(monkeypatch):
    """모델 파일 없이, 가짜 엔진과 토크나이저로 실제 번역기 클래스를 만든다."""
    holder = {}
    monkeypatch.setattr(translation, "_Ct2Model", lambda *args: holder["engine"])
    monkeypatch.setattr(translation, "load_sentencepiece", lambda path: SplitProcessor())
    return lambda fake: holder.update(engine=fake)


# 기본 동작: 엔진이 돌려준 조각을 목표 언어 SentencePiece로 이어 번역문으로 돌려준다.
def test_marian_returns_the_decoded_translation(engine):
    engine(FakeEngine(result=["Hello", "world"]))
    translator = translation.MarianTranslator(Path("model"), "ko", "en")
    assert translator.translate("안녕", "ko", "en") == "Hello world"


# Marian 엔진의 예외(CUDA 메모리 부족 RuntimeError)와 빈 결과가 모두 ModelError가 되는지 확인한다(커밋
# df874b5).
@pytest.mark.parametrize(
    "fake", [FakeEngine(error=RuntimeError("CUDA out of memory")), FakeEngine(result=[])]
)
def test_marian_engine_failures_become_model_errors(engine, fake):
    engine(fake)
    with pytest.raises(ModelError):
        translation.MarianTranslator(Path("model"), "ko", "en").translate("안녕", "ko", "en")


# NLLB도 같다. NLLB는 결과 맨 앞의 목표 언어 코드(eng_Latn)를 떼고 디코딩하므로, 코드만 돌아오면 빈 번역이라
# ModelError다.
@pytest.mark.parametrize(
    "fake", [FakeEngine(error=RuntimeError("CUDA out of memory")), FakeEngine(result=["eng_Latn"])]
)
def test_nllb_engine_failures_become_model_errors(engine, fake):
    engine(fake)
    with pytest.raises(ModelError):
        translation.NllbTranslator(Path("model")).translate("안녕", "ko", "en")


# OPUS-MT 학습 때의 preprocess.sh를 파이썬으로 옮긴 전처리: 전각 문장부호·둥근 따옴표를 ASCII로, 동양식
# 마침표(。) 뒤에 공백을, 폭 없는 공백·탭 같은 제어 문자는 공백으로 바꾸고 공백을 하나로 줄인다. 학습 때와
# 같은 글자를 넣어야 모델이 본 분포와 맞는다.
def test_opus_preprocess_follows_the_training_script():
    assert opus_preprocess("“안녕”，  세상！") == '"안녕", 세상!'
    assert opus_preprocess("끝。다음") == "끝. 다음"
    assert opus_preprocess(" a​b\tc ") == "a b c"  # 폭 없는 문자와 제어 문자는 공백이 된다


# 문장 나누기 사례: 문장 끝 부호(. ? ! 。 ？ ！) 뒤 공백·줄바꿈에서 나누되, 약어(U.N., Mr., e.g.)와 이니셜(J.)
# 뒤에서는 나누지 않고, 3.5처럼 공백이 없는 숫자는 건드리지 않는다. 약어가 줄바꿈이나 탭 뒤에 와도 약어로
# 알아보는지(커밋 05e16bb)와, 공백뿐인 입력은 빈 목록인지도 본다.
@pytest.mark.parametrize(
    ("text", "sentences"),
    [
        ("비가 왔다. 그래서 집에 있었다.", ["비가 왔다.", "그래서 집에 있었다."]),
        ("Is it? Yes!\nGood.", ["Is it?", "Yes!", "Good."]),
        ("첫째。 둘째？ 셋째！", ["첫째。", "둘째？", "셋째！"]),
        (
            "The U.N. met Mr. Smith and J. Doe, e.g. at 3.5 km. Then left.",
            ["The U.N. met Mr. Smith and J. Doe, e.g. at 3.5 km.", "Then left."],
        ),
        ("  한 문장  ", ["한 문장"]),
        ("Meet\nDr. Kim. Next.", ["Meet\nDr. Kim.", "Next."]),
        ("Ask\tMr. Lee first. Then go.", ["Ask\tMr. Lee first.", "Then go."]),
        ("   ", []),
    ],
)
def test_split_sentences(text, sentences):
    assert split_sentences(text) == sentences


# by_sentence를 켜면 세 문장을 따로 엔진에 넣고(호출 3번) 결과를 공백 하나로 잇는다. opus-mt가 한 문장씩
# 학습돼 여러 문장 입력에서 문장을 빠뜨렸기 때문에 영→한에 쓰는 방식이다(README 설계 판단).
def test_marian_by_sentence_translates_each_sentence_on_its_own(engine):
    echo = EchoEngine()
    engine(echo)
    translator = translation.MarianTranslator(Path("model"), "en", "ko", by_sentence=True)
    result = translator.translate("It rained. Dr. Kim left early!  Why?", "en", "ko")
    assert result == "IT RAINED. DR. KIM LEFT EARLY! WHY?"
    assert len(echo.calls) == 3


# 실제 Marian 어댑터를 거쳐 엔진에 들어가는 토큰을 문장별로 확인한다(커밋 7c972a1). 줄바꿈·탭 뒤의 약어(Dr.,
# Mr., J.)가 문장을 끊지 않아야 하고, 각 문장 끝에 끝 토큰 </s>가 붙는다. 전처리가 줄바꿈은 남기고 탭은
# 공백으로 바꾸므로 두 경우를 모두 본다.
@pytest.mark.parametrize(
    ("text", "calls"),
    [
        ("Meet\nDr. Kim. Next.", [["Meet", "Dr.", "Kim.", "</s>"], ["Next.", "</s>"]]),
        ("Ask\tMr. Lee first. Then go.", [["Ask", "Mr.", "Lee", "first.", "</s>"], ["Then", "go.", "</s>"]]),
        ("Met\nJ. Doe. Bye.", [["Met", "J.", "Doe.", "</s>"], ["Bye.", "</s>"]]),
    ],
)
def test_marian_by_sentence_keeps_abbreviations_after_any_white_space(engine, text, calls):
    echo = EchoEngine()
    engine(echo)
    translation.MarianTranslator(Path("model"), "en", "ko", by_sentence=True).translate(text, "en", "ko")
    assert echo.calls == calls


# 기본값(by_sentence 끔)은 입력 전체를 한 번에 번역한다. 한→영은 나눠도 차이가 없어 통째로 번역한다(README).
def test_marian_translates_the_whole_input_by_default(engine):
    echo = EchoEngine()
    engine(echo)
    translation.MarianTranslator(Path("model"), "en", "ko").translate("One. Two.", "en", "ko")
    assert len(echo.calls) == 1


# 엔진 입력 토큰 형식: Marian은 전처리(전각 쉼표 → ',')한 조각 뒤에 </s>, NLLB는 맨 앞에 원문 언어
# 코드(kor_Hang)와 끝에 </s>를 붙인다.
def test_source_tokens_add_language_code_and_end_of_sentence():
    assert marian_source_tokens(SplitProcessor(), "Hello，world") == ["Hello,world", "</s>"]
    assert nllb_source_tokens(SplitProcessor(), "안녕 세상", "ko") == ["kor_Hang", "안녕", "세상", "</s>"]


# 방향마다 다른 모델(opus는 한 방향 전용)을 연결하는 DirectionalTranslator: 등록된 방향은 그 모델로 넘기고
# 모델 이름에 방향을 적으며, 없는 방향·같은 언어·빈 글자는 ModelError로 거절한다.
def test_directional_translator_routes_each_direction():
    translator = DirectionalTranslator({("ko", "en"): FakeTranslator()})
    assert translator.translate("안녕", "ko", "en") == "[ko->en] 안녕"
    assert "ko->en" in translator.model_name
    with pytest.raises(ModelError, match="no translation model"):
        translator.translate("hello", "en", "ko")
    with pytest.raises(ModelError):
        translator.translate("안녕", "ko", "ko")
    with pytest.raises(ModelError):
        translator.translate("   ", "ko", "en")


# 보낸 요청 본문을 기록하고 handler의 응답을 돌려주는 가짜 Ollama 서버에 연결된 OllamaTranslator(비교용 LLM
# 번역기)를 만든다. httpx.MockTransport라 실제 네트워크를 쓰지 않는다.
def ollama_with(handler) -> tuple[OllamaTranslator, list[dict]]:
    sent: list[dict] = []

    def record(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return handler(request)

    translator = OllamaTranslator("qwen2.5:7b-instruct")
    translator._client = httpx.Client(base_url="http://ollama.test", transport=httpx.MockTransport(record))
    return translator, sent


# 고정된 시스템 프롬프트(한국어 → 영어 번역만 답하라)와 temperature 0, 스트리밍 없음으로 요청하고, 답의 앞뒤
# 공백을 지워 돌려주는지 확인한다. 측정할 때와 같은 결정적 설정이어야 결과를 재현할 수 있다.
def test_ollama_translator_uses_a_fixed_prompt_and_returns_the_reply():
    translator, sent = ollama_with(lambda _: httpx.Response(200, json={"message": {"content": " Hello \n"}}))
    assert translator.translate("안녕하세요", "ko", "en") == "Hello"
    payload = sent[0]
    assert payload["options"]["temperature"] == 0
    assert payload["stream"] is False
    assert "Korean text into English" in payload["messages"][0]["content"]
    assert payload["messages"][1] == {"role": "user", "content": "안녕하세요"}


# 서버 오류(500), 공백뿐인 답, 글자가 아닌 content, message 없음, JSON이 아닌 본문이 모두 ModelError가 되는지
# 확인한다.
@pytest.mark.parametrize(
    "reply",
    [
        httpx.Response(500, json={"error": "model crashed"}),
        httpx.Response(200, json={"message": {"content": "   "}}),
        httpx.Response(200, json={"message": {"content": None}}),
        httpx.Response(200, json={"message": {"content": 123}}),
        httpx.Response(200, json={"message": None}),
        httpx.Response(200, text="not json"),
    ],
)
def test_ollama_failures_become_model_errors(reply):
    translator, _ = ollama_with(lambda _: reply)
    with pytest.raises(ModelError):
        translator.translate("hello", "en", "ko")


# Ollama에 연결할 수 없을 때도 ModelError다.
def test_ollama_connection_error_becomes_model_error():
    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    translator, _ = ollama_with(refuse)
    with pytest.raises(ModelError):
        translator.translate("hello", "en", "ko")


# 여기부터 Ollama가 제공하는 Hy-MT2 번역기(T83, docs/experiments.md 15절)

# 모델 카드의 기본 번역 프롬프트(목표 언어가 Korean일 때). 요청 본문과 글자 하나까지 비교할 기대값이다.
HY_MT_ENGLISH_PROMPT = (
    "Translate the following text into Korean. Note that you should only output the translated result "
    "without any additional explanation:\n\n"
)


# HyMtTranslator를 가짜 Ollama 서버에 연결해 만든다. 요청마다 (경로, 본문)을 기록해 prepare가 어떤 순서로
# 무엇을 보냈는지도 볼 수 있게 한다.
def hymt_with(handler, **kwargs) -> tuple[HyMtTranslator, list[tuple[str, dict | None]]]:
    sent: list[tuple[str, dict | None]] = []

    def record(request: httpx.Request) -> httpx.Response:
        sent.append((request.url.path, json.loads(request.content) if request.content else None))
        return handler(request)

    translator = HyMtTranslator("hy-mt2:1.8b-q8_0", base_url="http://ollama.test", **kwargs)
    translator._client = httpx.Client(base_url="http://ollama.test", transport=httpx.MockTransport(record))
    return translator, sent


# Ollama /api/chat의 정상 응답 모양(끝난 이유, 입력·출력 토큰 수 포함)을 만든다.
def chat_reply(content, done_reason="stop", **extra) -> httpx.Response:
    body = {
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": done_reason,
        "prompt_eval_count": 40,
        "eval_count": 7,
        **extra,
    }
    return httpx.Response(200, json=body)


# Hy-MT2 요청 형식이 측정 전에 정한 그대로인지 확인한다: 시스템 프롬프트 없이 모델 카드 프롬프트 + 원문을
# 사용자 메시지 하나로, 스트리밍 없음, keep_alive -1(모델을 내리지 않음), 옵션은 탐욕 디코딩·카드의 반복 벌점
# 1.05·opus와 같은 출력 상한 256토큰·고정 문맥 길이 2048. 모델 이름은 'ollama/<모델>'이다.
def test_hymt_sends_the_model_card_prompt_as_the_only_message():
    translator, sent = hymt_with(lambda _: chat_reply(" 안녕하세요. \n"))
    assert translator.translate("Hello.", "en", "ko") == "안녕하세요."
    path, payload = sent[0]
    assert path == "/api/chat"
    # 모델 카드의 "Default Translation" 영어 프롬프트. 모델에 기본 시스템 프롬프트가 없어 넣지 않는다.
    assert payload["messages"] == [{"role": "user", "content": HY_MT_ENGLISH_PROMPT + "Hello."}]
    assert payload["model"] == "hy-mt2:1.8b-q8_0"
    assert payload["stream"] is False
    assert payload["keep_alive"] == -1
    # 탐욕 디코딩, 카드의 반복 벌점, opus와 같은 출력 상한 256토큰, 고정 문맥 길이.
    assert payload["options"] == {
        "temperature": 0,
        "repeat_penalty": 1.05,
        "num_predict": 256,
        "num_ctx": 2048,
    }
    assert translator.model_name == "ollama/hy-mt2:1.8b-q8_0"


# 한→영이면 프롬프트의 목표 언어가 English이고, 원문은 프롬프트 끝(빈 줄 뒤)에 붙는다.
def test_hymt_names_the_target_language_in_english():
    translator, sent = hymt_with(lambda _: chat_reply("Hello."))
    translator.translate("안녕하세요.", "ko", "en")
    assert sent[0][1]["messages"][0]["content"].startswith("Translate the following text into English. ")
    assert sent[0][1]["messages"][0]["content"].endswith(":\n\n안녕하세요.")


# 문장 단위 팔(by_sentence)은 opus 영→한처럼 문장마다 따로 요청하고 결과를 공백으로 잇는다. 모델 이름 끝의
# /split으로 통째로 팔과 구분한다.
def test_hymt_by_sentence_translates_each_sentence_on_its_own():
    replies = iter(["안녕.", "잘 지내?"])
    translator, sent = hymt_with(lambda _: chat_reply(next(replies)), by_sentence=True)
    assert translator.translate("Hi there. How are you?", "en", "ko") == "안녕. 잘 지내?"
    assert [payload["messages"][0]["content"] for _, payload in sent] == [
        HY_MT_ENGLISH_PROMPT + "Hi there.",
        HY_MT_ENGLISH_PROMPT + "How are you?",
    ]
    assert translator.model_name == "ollama/hy-mt2:1.8b-q8_0/split"


# calls에는 마지막 번역의 요청별 정보(끝난 이유, 입력·출력 토큰 수)만 남는다. 평가 도구가 번역 하나의 세부를
# 읽으므로 이전 번역의 기록이 섞이면 안 된다.
def test_hymt_keeps_the_details_of_the_last_translation_for_evaluation():
    translator, _ = hymt_with(lambda _: chat_reply("안녕.", eval_count=3), by_sentence=True)
    translator.translate("Hi. Bye.", "en", "ko")
    translator.translate("Hi.", "en", "ko")
    assert translator.calls == [{"done_reason": "stop", "prompt_eval_count": 40, "eval_count": 3}]


# 출력 상한에서 잘린 답(done_reason "length")도 버리지 않고 번역으로 쓰고, 잘렸다는 사실은 calls에 남긴다.
# opus가 최대 디코딩 길이 256에서 잘려도 번역으로 쓰는 것과 같게 맞춘 것이다.
def test_hymt_keeps_a_reply_cut_at_the_output_cap():
    translator, _ = hymt_with(lambda _: chat_reply("아주 긴 번역", done_reason="length"))
    assert translator.translate("A very long text.", "en", "ko") == "아주 긴 번역"
    assert translator.calls[0]["done_reason"] == "length"


# 서버 오류, 모델 없음(404), 빈 답과 잘못된 응답 모양이 모두 ModelError가 되는지 확인한다.
@pytest.mark.parametrize(
    "reply",
    [
        httpx.Response(500, json={"error": "model crashed"}),
        httpx.Response(404, json={"error": "model not found"}),
        chat_reply("   "),
        chat_reply(None),
        chat_reply(123),
        httpx.Response(200, json={"message": None}),
        httpx.Response(200, text="not json"),
    ],
)
def test_hymt_failures_become_model_errors(reply):
    translator, _ = hymt_with(lambda _: reply)
    with pytest.raises(ModelError):
        translator.translate("hello", "en", "ko")


# Ollama가 꺼져 있으면 ModelError다.
def test_hymt_connection_error_becomes_model_error():
    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    translator, _ = hymt_with(refuse)
    with pytest.raises(ModelError):
        translator.translate("hello", "en", "ko")


# prepare가 부르는 세 경로를 흉내 내는 가짜 Ollama: /api/show(모델 정보, 상태 코드 지정 가능),
# /api/tags(설치된 모델과 빌드 digest 목록), /api/generate(모델 적재).
def ollama_server(tags_digest="sha256-of-the-build", show_status=200):
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/show":
            return httpx.Response(show_status, json={"template": "..."})
        if request.url.path == "/api/tags":
            models = [
                {"name": "other:1b", "digest": "other"},
                {"name": "hy-mt2:1.8b-q8_0", "digest": tags_digest},
            ]
            return httpx.Response(200, json={"models": models})
        if request.url.path == "/api/generate":
            return httpx.Response(200, json={"done": True})
        return httpx.Response(404)

    return handle


# prepare가 목록에서 이 모델의 digest를 찾아 돌려주고 기억하며, 프롬프트 없는 generate 요청에 keep_alive -1을
# 주어 모델을 올려 둔 채로 두는지 확인한다(첫 번역 요청이 적재 시간을 내지 않게).
def test_hymt_prepare_loads_the_model_to_stay_and_reports_its_build():
    translator, sent = hymt_with(ollama_server())
    assert translator.prepare() == "sha256-of-the-build"
    assert translator.digest == "sha256-of-the-build"
    generate = [payload for path, payload in sent if path == "/api/generate"]
    assert generate == [{"model": "hy-mt2:1.8b-q8_0", "keep_alive": -1}]


# 회귀 테스트(커밋 605eed4): 시작 대기 시간(HYMT_PREPARE_TIMEOUT_S)의 남은 시간을 넘기면 prepare 안의 어떤
# 요청도 그보다 오래 기다리지 않는다. timeout_s를 주지 않으면 확인 요청(show·tags)은 요청 한도(30초), 적재는
# 적재 한도(120초)를 쓰고, timeout_s=5를 주면 세 요청 모두 약 5초로 줄어든다. 요청마다 httpx에 넘긴 read
# 제한을 기록해 확인한다.
def test_hymt_prepare_keeps_each_request_within_the_time_left():
    server, limits = ollama_server(), {}

    def handle(request: httpx.Request) -> httpx.Response:
        limits[request.url.path] = request.extensions["timeout"]["read"]
        return server(request)

    translator, _ = hymt_with(handle, timeout_s=30, prepare_timeout_s=120)
    translator.prepare()
    assert limits == {"/api/show": 30, "/api/tags": 30, "/api/generate": 120}
    translator.prepare(timeout_s=5)
    assert limits == pytest.approx({"/api/show": 5, "/api/tags": 5, "/api/generate": 5}, abs=0.5)


# 회귀 테스트(커밋 aeaf53e): 남은 시간을 prepare 시작 때 한 번만 재서 세 요청에 똑같이 주면 한 시도가 남은
# 시간의 약 3배까지 걸릴 수 있었다. 이제 요청마다 그 순간의 남은 시간을 다시 잰다. time.monotonic을 가짜
# 시계로 바꾸고 show·tags가 4초씩 쓰게 하면 10초 중 10 → 6 → 2초가 된다. 남은 시간이 없어도 적재 요청은 0이나
# 음수가 아닌 아주 작은 제한(최소 0.01초)을 받는다.
def test_hymt_prepare_takes_the_time_left_again_before_each_request(monkeypatch):
    server, limits, clock = ollama_server(), {}, [100.0]

    def handle(request: httpx.Request) -> httpx.Response:
        limits[request.url.path] = request.extensions["timeout"]["read"]
        if request.url.path in ("/api/show", "/api/tags"):
            clock[0] += 4
        return server(request)

    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    translator, _ = hymt_with(handle, timeout_s=30, prepare_timeout_s=120)
    translator.prepare(timeout_s=10)
    assert limits == pytest.approx({"/api/show": 10, "/api/tags": 6, "/api/generate": 2})
    clock[0] = 100.0
    limits.clear()
    translator.prepare(timeout_s=7)  # 적재에 남은 시간이 없어도 유효한 아주 작은 제한을 받는다
    assert 0 < limits["/api/generate"] <= 0.01


# Ollama에 모델이 없으면(show 404) 만드는 방법(eval.hymt_setup)을 알려 주는 ModelError로 멈추고, 적재 요청은
# 보내지 않는다.
def test_hymt_prepare_fails_when_ollama_lacks_the_model():
    translator, sent = hymt_with(ollama_server(show_status=404))
    with pytest.raises(ModelError, match="hymt_setup"):
        translator.prepare()
    assert not any(path == "/api/generate" for path, _ in sent)


# Ollama가 아직 떠 있지 않으면 ModelError다. 기다릴 가치가 있는 실패라 시작 쪽이 다시 시도할 수 있게
# RuntimeError와 구분한다(prepare docstring).
def test_hymt_prepare_fails_while_ollama_is_down():
    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    translator, _ = hymt_with(refuse)
    with pytest.raises(ModelError):
        translator.prepare()


# 설치된 모델의 digest가 측정한 빌드와 다르면 RuntimeError로 멈추고 적재하지 않는다. 같은 이름이라도 다른
# 빌드면 측정 결과가 들어맞지 않으므로, 기다려도 풀리지 않는 실패로 따로 다룬다.
def test_hymt_prepare_refuses_another_build_of_the_model():
    translator, sent = hymt_with(
        ollama_server(tags_digest="another-build"), expected_digest="the-measured-build"
    )
    with pytest.raises(RuntimeError, match="another-build"):
        translator.prepare()
    assert not any(path == "/api/generate" for path, _ in sent)

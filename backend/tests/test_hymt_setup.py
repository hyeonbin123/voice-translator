# [파일 역할] Hy-MT2를 Ollama에 올린 뒤 기록한 빌드인지 보는 eval/hymt_setup.py의 check(docs/experiments.md
# 15절 0단계)가 모델 목록에서 digest를 서버와 같은 기준으로 찾는지 확인한다. Ollama는 httpx MockTransport로
# 바꾸고 GPU·모델은 쓰지 않는다.
# [왜 중요한가] 서버(app/services/translation.py의 HyMtTranslator.prepare)는 /api/tags 목록에서 name 또는
# model 필드가 이름과 같은 모델의 digest를 쓴다. 점검 도구가 다른 기준으로 찾으면 같은 Ollama를 두고
# 서버와 점검 결과가 갈린다.

import httpx
import pytest

from eval import hymt_setup

NAME, DIGEST = hymt_setup.QUANTS["q8_0"][3], hymt_setup.QUANTS["q8_0"][4]


# 기록한 빌드처럼 답하는 가짜 Ollama. entry는 /api/tags 목록의 한 항목이다.
def fake_ollama(monkeypatch, entry: dict) -> None:
    show = {
        "template": hymt_setup.TEMPLATE,
        "parameters": "\n".join(f'stop "{stop}"' for stop in hymt_setup.STOPS),
        "model_info": {"general.architecture": hymt_setup.ARCHITECTURE},
    }

    def answer(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/show":
            return httpx.Response(200, json=show)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [entry]})
        return httpx.Response(200, json={"version": "0.35.1"})

    client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: client(transport=httpx.MockTransport(answer), **kw))


# 목록의 model 필드만 이름과 같아도(name은 다르게 적힘) 서버처럼 그 모델의 digest를 찾아 통과한다. 예전에는
# name만 봐서 digest를 못 찾고 멈췄다.
def test_check_finds_the_digest_by_name_or_model_as_the_server_does(monkeypatch):
    fake_ollama(monkeypatch, {"name": f"{NAME}-copy", "model": NAME, "digest": DIGEST})
    result = hymt_setup.check("q8_0", "http://ollama.test")
    assert result["digest"] == DIGEST and result["problems"] == []


# 이름이 어느 필드에도 없으면 digest가 없어 멈춘다.
def test_check_stops_when_ollama_does_not_list_the_model(monkeypatch):
    fake_ollama(monkeypatch, {"name": "other:1b", "model": "other:1b", "digest": DIGEST})
    with pytest.raises(SystemExit):
        hymt_setup.check("q8_0", "http://ollama.test")

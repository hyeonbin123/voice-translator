# [파일 역할] 번역 기록 API(GET /api/history, GET·DELETE /api/history/{id}, routers/history.py)를 실제 테스트
# DB(PostgreSQL)로 검증한다.
# [무엇을 확인하나] 최신순 정렬과 페이지 나누기, 상세 조회, 삭제 시 번역 음성 행이 함께 지워지는지, 다른
# 사용자의 기록은 존재 여부조차 드러내지 않고 404인지, 잘못된 쿼리·id와 인증 없는 요청이 거절되는지.
# [왜 중요한가] 기록은 사용자별 개인 데이터라, 소유자 조건(user_id)이 빠지는 회귀는 곧 정보 노출이다.

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

# ORM 모델로 DB에 직접 기록을 넣고, 삭제 뒤 행이 남았는지 DB에서 직접 확인한다(API 응답만 보지 않으려는 것).
from app.models import AudioFile, Translation, User


# 로그인 사용자(user)의 기록 3개와 다른 사용자(other)의 기록 1개를 만든다. 두 기록은 같은 시각(minutes=1)이라
# created_at이 같을 때 id로 순서를 정하는 규칙까지 시험할 수 있다.
# 첫 기록과 다른 사용자의 기록에는 번역 음성 행(AudioFile)을 붙인다. path는 실제 파일이 없는 값이다. 음성
# 저장소의 delete는 없는 파일을 그냥 넘어가므로(services/audio_store.py) 삭제 테스트에 실제 파일이 필요 없다.
# flush로 id를 먼저 받아 AudioFile의 외래 키에 쓰고, 마지막에 한 번 commit한다.
@pytest.fixture
async def histories(db_session, user, password_hash):
    other = User(email="other@example.com", hashed_password=password_hash)
    db_session.add(other)
    await db_session.flush()
    now = datetime.now(UTC)
    rows = []
    for owner, minutes in [(user, 0), (user, 1), (user, 1), (other, 2)]:
        row = Translation(
            user_id=owner.id,
            mode="text",
            source_lang="ko",
            target_lang="en",
            source_text="안녕하세요",
            translated_text="Hello",
            mt_model="test-mt",
            mt_ms=12,
            created_at=now + timedelta(minutes=minutes),
        )
        rows.append(row)
        db_session.add(row)
    await db_session.flush()
    audio = AudioFile(translation_id=rows[0].id, path="private/generated.wav", duration_ms=1000)
    other_audio = AudioFile(translation_id=rows[-1].id, path="private/other.wav", duration_ms=1000)
    db_session.add_all([audio, other_audio])
    await db_session.commit()
    return rows, audio, other_audio


# 기록이 없으면 404가 아니라 빈 목록과 total 0으로 답하는지 확인한다(화면이 빈 상태를 그대로 그릴 수 있게).
async def test_empty_history(client, auth_headers):
    response = await client.get("/api/history", headers=auth_headers)
    assert response.status_code == 200
    assert response.json() == {"items": [], "total": 0}


# 목록은 내 기록만(total 3), created_at 내림차순이고 같은 시각이면 id 내림차순이어야 한다(라우터의 order_by).
# 정렬 기준에 id가 없으면 같은 시각의 두 기록이 페이지를 넘길 때 겹치거나 빠질 수 있다.
# limit·offset이 같은 순서에서 잘라 내는지, 범위를 넘은 offset은 빈 목록인지, 상세 응답이 목록 항목과 같은지
# 본다. 응답에 저장 경로(path)·user_id 같은 내부 값이 없고, 음성이 있는 기록만 audio_id가 채워지는지도
# 확인한다.
async def test_history_pagination_order_and_detail(client, auth_headers, histories):
    rows, audio, _ = histories
    expected = sorted(rows[:3], key=lambda row: (row.created_at, row.id), reverse=True)
    response = await client.get("/api/history", headers=auth_headers)
    assert response.status_code == 200
    page = response.json()
    assert page["total"] == 3
    assert [item["id"] for item in page["items"]] == [str(row.id) for row in expected]
    response = await client.get("/api/history?limit=1&offset=1", headers=auth_headers)
    assert response.json() == {"total": 3, "items": [page["items"][1]]}
    assert (await client.get("/api/history?offset=99", headers=auth_headers)).json() == {
        "total": 3,
        "items": [],
    }
    for item in page["items"]:
        detail = await client.get(f"/api/history/{item['id']}", headers=auth_headers)
        assert detail.status_code == 200 and detail.json() == item
        assert item["source_text"] == "안녕하세요" and item["translated_text"] == "Hello"
        assert item["audio_id"] == (str(audio.id) if item["id"] == str(rows[0].id) else None)
        assert "path" not in item and "user_id" not in item
        assert item["stt_ms"] is None and item["mt_ms"] == 12


# 삭제하면 204와 빈 본문을 주고, DB에서 번역 행과 그 음성 행(외래 키 ondelete=CASCADE)이 함께 사라지며, 다른
# 사용자의 음성 행은 남는지 확인한다. 지운 기록은 다시 조회·삭제해도 404다.
# expunge_all은 세션의 식별자 캐시를 비워, get이 캐시된 객체가 아니라 실제 DB를 다시 읽게 하려는 것이다.
async def test_delete_cascades_audio_rows_and_preserves_other_history(
    client, auth_headers, histories, db_session
):
    rows, audio, other_audio = histories
    deleted_id, audio_id, other_id = rows[0].id, audio.id, other_audio.id
    response = await client.delete(f"/api/history/{deleted_id}", headers=auth_headers)
    assert response.status_code == 204 and response.content == b""
    db_session.expunge_all()
    assert await db_session.get(Translation, deleted_id) is None
    assert await db_session.get(AudioFile, audio_id) is None
    assert await db_session.get(AudioFile, other_id) is not None
    assert (await client.get(f"/api/history/{deleted_id}", headers=auth_headers)).status_code == 404
    assert (await client.delete(f"/api/history/{deleted_id}", headers=auth_headers)).status_code == 404
    page = (await client.get("/api/history", headers=auth_headers)).json()
    assert page["total"] == 2
    # 음성이 없는 기록도 똑같이 지울 수 있어야 한다.
    no_audio_id = page["items"][0]["id"]
    assert (await client.delete(f"/api/history/{no_audio_id}", headers=auth_headers)).status_code == 204
    assert await db_session.get(Translation, UUID(no_audio_id)) is None


# 다른 사용자의 기록 id와 없는 id 모두 같은 404 "History not found"로 답하는지 확인한다(docs/api.md: 다른
# 사용자 기록과 없는 기록은 같은 404). 응답을 나누면 남의 기록이 있다는 사실이 드러난다. DELETE 요청이 남의
# 기록과 음성 행을 실제로 지우지 않았는지도 DB에서 확인한다.
@pytest.mark.parametrize("method", ["get", "delete"])
async def test_foreign_and_missing_history_are_404(client, auth_headers, histories, db_session, method):
    rows, _, other_audio = histories
    request = getattr(client, method)
    for id in [rows[-1].id, uuid4()]:
        response = await request(f"/api/history/{id}", headers=auth_headers)
        assert response.status_code == 404 and response.json() == {"detail": "History not found"}
    db_session.expunge_all()
    assert await db_session.get(Translation, rows[-1].id) is not None
    assert await db_session.get(AudioFile, other_audio.id) is not None


# 페이지 매개변수 검증: limit은 1~100의 정수, offset은 0 이상 2147483647(2^31 - 1) 이하의 정수(라우터의
# Query(ge, le)). 범위 밖·문자열·소수는 422다. 너무 큰 limit으로 한 번에 많은 행을 읽는 요청을 막는다.
# offset 상한 바로 위(2^31)와 PostgreSQL bigint도 넘는 값(2^63)을 넣은 이유: 상한이 없을 때 2^31은 DB까지
# 내려가 200(빈 페이지)이 났고, 2^63은 DB 드라이버가 bigint로 보내지 못해 처리되지 않은 오류(500)가 났다.
# 둘 다 다른 범위 오류처럼 DB에 묻기 전에 422로 막혀야 한다.
@pytest.mark.parametrize(
    "query",
    [
        "limit=0",
        "limit=101",
        "offset=-1",
        "limit=abc",
        "offset=1.5",
        "offset=2147483648",
        "offset=9223372036854775808",
    ],
)
async def test_invalid_pagination(client, auth_headers, query):
    assert (await client.get(f"/api/history?{query}", headers=auth_headers)).status_code == 422


# offset 상한 값(2^31 - 1) 자체는 받아들여 실제 DB에서 오류 없이 빈 페이지를 돌려주는지 확인한다. total은
# 페이지와 상관없는 본인 기록 수(3)를 그대로 준다(docs/api.md: offset이 끝을 넘으면 items만 빈 배열). 상한을
# PostgreSQL integer 범위로 둔 것은 이 값까지는 쿼리 인자가 bigint로 바뀌지 않기 때문이다(routers/history.py).
async def test_the_largest_offset_is_an_empty_page(client, auth_headers, histories):
    response = await client.get("/api/history?offset=2147483647", headers=auth_headers)
    assert response.status_code == 200
    assert response.json() == {"items": [], "total": 3}


# 기록 API의 세 경로 모두 인증 헤더가 없으면 401이다.
@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/api/history"),
        ("get", f"/api/history/{uuid4()}"),
        ("delete", f"/api/history/{uuid4()}"),
    ],
)
async def test_history_requires_auth(client, method, path):
    assert (await getattr(client, method)(path)).status_code == 401


# id가 UUID 형식이 아니면 DB에 묻기 전에 경로 검증에서 422로 거절되는지 확인한다.
@pytest.mark.parametrize("method", ["get", "delete"])
async def test_invalid_history_id(client, auth_headers, method):
    assert (await getattr(client, method)("/api/history/not-a-uuid", headers=auth_headers)).status_code == 422

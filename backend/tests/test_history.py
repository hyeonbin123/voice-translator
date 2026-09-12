from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from app.models import AudioFile, Translation, User


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


async def test_empty_history(client, auth_headers):
    response = await client.get("/api/history", headers=auth_headers)
    assert response.status_code == 200
    assert response.json() == {"items": [], "total": 0}


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
    # Deleting a record without audio must work as well.
    no_audio_id = page["items"][0]["id"]
    assert (await client.delete(f"/api/history/{no_audio_id}", headers=auth_headers)).status_code == 204
    assert await db_session.get(Translation, UUID(no_audio_id)) is None


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


@pytest.mark.parametrize("query", ["limit=0", "limit=101", "offset=-1", "limit=abc", "offset=1.5"])
async def test_invalid_pagination(client, auth_headers, query):
    assert (await client.get(f"/api/history?{query}", headers=auth_headers)).status_code == 422


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


@pytest.mark.parametrize("method", ["get", "delete"])
async def test_invalid_history_id(client, auth_headers, method):
    assert (await getattr(client, method)("/api/history/not-a-uuid", headers=auth_headers)).status_code == 422

# [파일 역할] 회원 가입·로그인·토큰 갱신·내 정보(/api/auth/*) API의 계약을 검증하는 테스트 묶음이다. 실제
# PostgreSQL 테스트 DB와 앱을 conftest.py의 fixture(client, db_session, user)로 쓴다.
# [무엇을 지키나] docs/api.md의 인증 규칙: 이메일은 소문자로 저장·비교, 비밀번호는 8자 이상·UTF-8 72바이트
# 이하(bcrypt가 72바이트까지만 쓰기 때문), access 30분·refresh 7일, 두 토큰은 서로 대신 쓸 수 없음, 실패는
# 이유를 가리지 않고 401과 WWW-Authenticate: Bearer, 토큰 응답은 Cache-Control: no-store.

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

# bcrypt: 저장된 해시가 실제로 그 비밀번호의 bcrypt 해시인지 확인. jwt: 발급된 토큰을 풀어 보고, 일부러
# 망가뜨린 토큰을 만든다. sqlalchemy: DB에 남은 사용자 수 확인과 사용자 삭제.
import bcrypt
import jwt
import pytest
from sqlalchemy import delete, func, select

from app.config import get_settings
from app.core.security import create_token
from app.models import User


# 정상 흐름 전체: 가입 → 로그인 → 내 정보 → 갱신 → 새 토큰으로 내 정보.
# 확인하는 것: 대소문자가 섞인 이메일이 소문자로 저장되고 로그인도 대소문자와 무관함, 응답에 해시 등 다른
# 필드가 없음, 비밀번호가 평문이 아니라 bcrypt 해시로 저장됨, 토큰의 종류(type)와 수명(30분·7일), no-store
# 헤더, 갱신하면 두 토큰 모두 새 문자열(jti가 토큰마다 달라 같은 초에도 다름).
# 막는 회귀: 비밀번호 평문 저장, 응답으로 해시 노출, 수명 설정 실수.
async def test_register_login_refresh_me(client, db_session):
    registered = await client.post(
        "/api/auth/register",
        json={
            "email": "Person@EXAMPLE.COM",
            "password": "password123",
        },
    )
    assert registered.status_code == 201
    account = registered.json()
    assert set(account) == {"id", "email", "created_at"}
    assert account["email"] == "person@example.com"
    stored = await db_session.get(User, UUID(account["id"]))
    assert stored.hashed_password != "password123"
    assert bcrypt.checkpw(b"password123", stored.hashed_password.encode())
    login = await client.post(
        "/api/auth/login", data={"username": "PERSON@example.com", "password": "password123"}
    )
    assert login.status_code == 200
    assert login.headers["cache-control"] == "no-store"
    pair = login.json()
    assert pair["expires_in"] == 1800 and pair["token_type"] == "bearer"
    for kind, lifetime in [("access", 1800), ("refresh", 604800)]:
        payload = jwt.decode(
            pair[f"{kind}_token"], get_settings().jwt_secret_key.get_secret_value(), algorithms=["HS256"]
        )
        assert payload["type"] == kind
        assert payload["exp"] - payload["iat"] == lifetime
    me = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {pair['access_token']}"})
    assert me.status_code == 200 and me.json() == account
    refreshed = await client.post("/api/auth/refresh", json={"refresh_token": pair["refresh_token"]})
    assert refreshed.status_code == 200
    assert refreshed.json()["access_token"] != pair["access_token"]
    assert refreshed.json()["refresh_token"] != pair["refresh_token"]
    me = await client.get(
        "/api/auth/me", headers={"Authorization": f"Bearer {refreshed.json()['access_token']}"}
    )
    assert me.json() == account


# 가입 입력 검증: 잘못된 이메일, 8자 미만, 72바이트 초과(영문 73자, 한글 25자 = 75바이트)는 422이고 DB에
# 사용자가 생기지 않아야 한다. 한글은 UTF-8에서 3바이트라 글자 수가 아니라 바이트로 세는지 확인하는 경우다.
@pytest.mark.parametrize(
    "email,password",
    [
        ("invalid", "password123"),
        ("person@example.com", "short"),
        ("person@example.com", "a" * 73),
        ("person@example.com", "가" * 25),
    ],
)
async def test_register_validation(client, db_session, email, password):
    response = await client.post("/api/auth/register", json={"email": email, "password": password})
    assert response.status_code == 422
    assert await db_session.scalar(select(func.count()).select_from(User)) == 0


# 72바이트 경계: 정확히 72바이트(영문 72자, 한글 24자)는 가입·로그인 모두 통과해야 한다. 경계에서 하나 차이로
# 거절하는 실수를 막는다.
@pytest.mark.parametrize("password", ["a" * 72, "가" * 24])
async def test_password_72_byte_boundary(client, password):
    data = {"email": "boundary@example.com", "password": password}
    assert (await client.post("/api/auth/register", json=data)).status_code == 201
    assert (
        await client.post("/api/auth/login", data={"username": data["email"], "password": password})
    ).status_code == 200


# 대소문자만 다른 같은 이메일로 두 가입 요청을 동시에 보내면 하나는 201, 하나는 409이고 사용자는 하나만 남아야
# 한다. '먼저 조회하고 없으면 넣기'만으로는 동시 요청이 둘 다 통과할 수 있으므로, DB 유일 제약
# 위반(IntegrityError)을 409로 바꾸는 처리(routers/auth.py)를 검증한다.
async def test_concurrent_duplicate_registration(client, db_session):
    results = await asyncio.gather(
        *[
            client.post("/api/auth/register", json={"email": email, "password": "password123"})
            for email in ["same@example.com", "SAME@example.com"]
        ]
    )
    assert sorted(response.status_code for response in results) == [201, 409]
    assert await db_session.scalar(select(func.count()).select_from(User)) == 1


# 로그인 실패는 이유(틀린 비밀번호, 없는 계정, 이메일 형식 오류, 72바이트 초과)와 관계없이 모두 401과
# WWW-Authenticate: Bearer여야 한다. 이유를 나누면 계정이 있는지 알려 주게 되기 때문이다(docs/api.md 오류
# 응답).
@pytest.mark.parametrize(
    "username,password",
    [
        ("person@example.com", "incorrect"),
        ("missing@example.com", "password123"),
        ("invalid", "password123"),
        ("person@example.com", "x" * 73),
        ("person@example.com", "가" * 25),
    ],
)
async def test_login_failure(client, user, username, password):
    response = await client.post("/api/auth/login", data={"username": username, "password": password})
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


# 인증 정보가 없을 때: 헤더 없는 내 정보는 401, 로그인은 폼(username·password)이어야 해서 JSON 본문은 422,
# 갱신에 refresh_token이 없으면 422.
async def test_missing_credentials(client):
    assert (await client.get("/api/auth/me")).status_code == 401
    assert (await client.post("/api/auth/login", json={"email": "person@example.com"})).status_code == 422
    assert (await client.post("/api/auth/refresh", json={})).status_code == 422


# 잘못된 토큰 7가지 × 두 경로(access로 내 정보, refresh로 갱신) = 14경우 모두 401이어야 한다: 형식이 아님,
# 만료, 다른 키로 서명, 종류가 반대(access와 refresh를 서로 대신 쓰기), sub가 UUID가 아님, exp 없음, 없는
# 사용자.
# 정상 토큰과 같은 클레임을 직접 만든 뒤 한 가지씩만 망가뜨려, 각 검증(서명·만료·필수 클레임·종류·사용자
# 존재)이 따로 동작하는지 본다.
@pytest.mark.parametrize("endpoint,kind", [("/api/auth/me", "access"), ("/api/auth/refresh", "refresh")])
@pytest.mark.parametrize(
    "fault", ["malformed", "expired", "signature", "type", "subject", "missing_exp", "unknown_user"]
)
async def test_invalid_tokens(client, user, endpoint, kind, fault):
    now = datetime.now(UTC)
    payload = {
        "sub": str(user.id),
        "type": kind,
        "iat": now,
        "exp": now + timedelta(minutes=5),
        "jti": str(uuid4()),
    }
    secret = get_settings().jwt_secret_key.get_secret_value()
    if fault == "expired":
        payload["exp"] = now - timedelta(seconds=1)
    elif fault == "signature":
        secret = "different-test-signing-key-32-bytes-minimum"
    elif fault == "type":
        payload["type"] = "refresh" if kind == "access" else "access"
    elif fault == "subject":
        payload["sub"] = "not-a-uuid"
    elif fault == "missing_exp":
        del payload["exp"]
    elif fault == "unknown_user":
        payload["sub"] = str(uuid4())
    token = "garbage" if fault == "malformed" else jwt.encode(payload, secret, algorithm="HS256")
    response = (
        await client.get(endpoint, headers={"Authorization": f"Bearer {token}"})
        if kind == "access"
        else await client.post(endpoint, json={"refresh_token": token})
    )
    assert response.status_code == 401


# 사용자를 지운 뒤에는 그 사용자의 유효한(서명·만료가 맞는) access·refresh 토큰도 401이어야 한다. 서명만 보고
# 통과시키면 탈퇴한 계정이 계속 쓰일 수 있으므로 사용자 존재까지 확인하는지 본다.
async def test_deleted_user_tokens_rejected(client, user, db_session):
    access, refresh = create_token(user.id, "access"), create_token(user.id, "refresh")
    await db_session.execute(delete(User).where(User.id == user.id))
    await db_session.commit()
    assert (
        await client.get("/api/auth/me", headers={"Authorization": f"Bearer {access}"})
    ).status_code == 401
    assert (await client.post("/api/auth/refresh", json={"refresh_token": refresh})).status_code == 401

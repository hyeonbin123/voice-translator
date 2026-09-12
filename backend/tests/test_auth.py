import asyncio
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import bcrypt
import jwt
import pytest
from sqlalchemy import delete, func, select

from app.config import get_settings
from app.core.security import create_token
from app.models import User


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


@pytest.mark.parametrize("password", ["a" * 72, "가" * 24])
async def test_password_72_byte_boundary(client, password):
    data = {"email": "boundary@example.com", "password": password}
    assert (await client.post("/api/auth/register", json=data)).status_code == 201
    assert (
        await client.post("/api/auth/login", data={"username": data["email"], "password": password})
    ).status_code == 200


async def test_concurrent_duplicate_registration(client, db_session):
    results = await asyncio.gather(
        *[
            client.post("/api/auth/register", json={"email": email, "password": "password123"})
            for email in ["same@example.com", "SAME@example.com"]
        ]
    )
    assert sorted(response.status_code for response in results) == [201, 409]
    assert await db_session.scalar(select(func.count()).select_from(User)) == 1


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


async def test_missing_credentials(client):
    assert (await client.get("/api/auth/me")).status_code == 401
    assert (await client.post("/api/auth/login", json={"email": "person@example.com"})).status_code == 422
    assert (await client.post("/api/auth/refresh", json={})).status_code == 422


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


async def test_deleted_user_tokens_rejected(client, user, db_session):
    access, refresh = create_token(user.id, "access"), create_token(user.id, "refresh")
    await db_session.execute(delete(User).where(User.id == user.id))
    await db_session.commit()
    assert (
        await client.get("/api/auth/me", headers={"Authorization": f"Bearer {access}"})
    ).status_code == 401
    assert (await client.post("/api/auth/refresh", json={"refresh_token": refresh})).status_code == 401

import asyncio
import os
from pathlib import Path
from uuid import uuid4

import asyncpg
import pytest
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from alembic import command
from app.config import get_settings
from app.core.security import create_token, hash_password
from app.dependencies import get_db
from app.main import app
from app.models import User

BACKEND = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def migrated_database():
    # This URL supplies host/admin credentials only. Never migrate or clear that database.
    admin_url = make_url(os.getenv("TEST_DATABASE_ADMIN_URL", get_settings().database_url))
    name = "vt_test_" + uuid4().hex
    test_url = admin_url.set(drivername="postgresql+asyncpg", database=name)

    async def manage_database(create):
        connection = await asyncpg.connect(
            admin_url.set(drivername="postgresql", database="postgres").render_as_string(hide_password=False)
        )
        try:
            # name is generated here, never taken from input or an environment variable.
            assert name.startswith("vt_test_") and len(name) == 40
            if create:
                await connection.execute(f'CREATE DATABASE "{name}"')
            else:
                await connection.execute(f'DROP DATABASE "{name}" WITH (FORCE)')
        finally:
            await connection.close()

    asyncio.run(manage_database(True))
    try:
        cfg = Config(str(BACKEND / "alembic.ini"))
        cfg.attributes["database_url"] = test_url
        command.upgrade(cfg, "head")
        yield test_url
    finally:
        asyncio.run(manage_database(False))


@pytest.fixture
async def db_engine(migrated_database):
    engine = create_async_engine(migrated_database)
    try:
        yield engine
    finally:
        async with engine.begin() as connection:
            await connection.execute(delete(User))
        await engine.dispose()


@pytest.fixture
async def db_session(db_engine):
    async with async_sessionmaker(db_engine, expire_on_commit=False)() as session:
        yield session


@pytest.fixture
async def client(db_engine):
    sessions = async_sessionmaker(db_engine, expire_on_commit=False)

    async def override_db():
        async with sessions() as session:
            yield session

    previous = app.dependency_overrides.copy()
    app.dependency_overrides[get_db] = override_db
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as api:
            yield api
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous)


@pytest.fixture(scope="session")
def password_hash():
    return hash_password("password123")


@pytest.fixture
async def user(db_session, password_hash):
    account = User(email="person@example.com", hashed_password=password_hash)
    db_session.add(account)
    await db_session.commit()
    return account


@pytest.fixture
def auth_headers(user):
    return {"Authorization": f"Bearer {create_token(user.id, 'access')}"}

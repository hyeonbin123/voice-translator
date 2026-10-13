# [파일 역할] 백엔드 pytest 공용 fixture. 테스트마다 실제 PostgreSQL에 만든 임시 DB, 그 DB를 쓰는 비동기 HTTP
# 클라이언트, 미리 만든 사용자와 인증 헤더를 제공한다.
# [왜 이렇게 하나] SQLite 같은 대체 DB가 아니라 compose의 PostgreSQL에 테스트 전용 DB(vt_test_<임의 문자열>)를
# 만들고 Alembic 마이그레이션으로 스키마를 올린다(README 테스트 절). 운영과 같은 DB 엔진·같은 마이그레이션으로
# 검증해야 방언 차이나 마이그레이션 누락을 잡을 수 있다. 끝나면 그 DB를 지운다.
# [흐름] migrated_database(세션당 한 번: DB 생성 → 마이그레이션 → 끝나면 삭제) → db_engine(테스트마다 엔진,
# 끝나면 사용자 행 삭제) → db_session·client(앱의 get_db를 테스트 DB 세션으로 바꿔 끼움) →
# user·auth_headers(로그인한 요청용).
# [연결] test_auth.py, test_history.py, test_translate_api.py, test_dialog_api.py, test_live_api.py,
# test_pipeline.py 등이 이 fixture를 쓴다. pyproject의 asyncio_mode = auto라 async fixture·테스트에 따로
# 표시를 달지 않는다.

import asyncio
import os
from pathlib import Path
from uuid import uuid4

# asyncpg: DB를 만들고 지우는 관리용 직접 연결. alembic: 테스트 DB에 마이그레이션 적용. httpx ASGITransport:
# 서버를 띄우지 않고 FastAPI 앱에 메모리 안에서 요청을 보낸다. sqlalchemy: 테스트 DB 엔진·세션.
import asyncpg
import pytest
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

# 앱 쪽에서 가져오는 것: 설정(기본 DB 주소), 토큰·비밀번호 해시 함수, 엔진 옵션(앱과 같은 옵션으로 테스트
# 엔진을 만듦), 바꿔 끼울 의존성 get_db, FastAPI 앱, User 모델.
from alembic import command
from app.config import get_settings
from app.core.security import create_token, hash_password
from app.db.session import ENGINE_OPTIONS
from app.dependencies import get_db
from app.main import app
from app.models import User

# backend/ 폴더 경로. alembic.ini를 찾는 데 쓴다.
BACKEND = Path(__file__).resolve().parents[1]


# 테스트 세션 전체에서 한 번: 임의 이름의 테스트 DB를 만들고 마이그레이션을 끝까지(head) 적용한 뒤 그 DB
# 주소를 돌려준다. 끝나면(실패해도 finally로) DB를 지운다.
# 관리 주소는 TEST_DATABASE_ADMIN_URL 또는 앱 기본 DB 주소에서 호스트·계정만 빌리고, 그 DB 자체는 절대
# 건드리지 않는다(아래 영어 주석). 이름에 uuid를 넣어 여러 실행이 같은 서버에서 겹쳐도 충돌하지 않는다.
@pytest.fixture(scope="session")
def migrated_database():
    # This URL supplies host/admin credentials only. Never migrate or clear that database.
    admin_url = make_url(os.getenv("TEST_DATABASE_ADMIN_URL", get_settings().database_url))
    name = "vt_test_" + uuid4().hex
    test_url = admin_url.set(drivername="postgresql+asyncpg", database=name)

    # 관리용 postgres DB에 접속해 테스트 DB를 만들거나(create=True) 지운다. CREATE/DROP DATABASE는 이름을
    # 바인딩 인자로 받을 수 없어 SQL 문자열에 넣으므로, 넣기 전에 이름이 이 함수가 만든 형식인지 assert로
    # 확인해 다른 DB를 지우는 일을 막는다. WITH (FORCE)는 남은 연결이 있어도 지운다(PostgreSQL 13 이상).
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

    # 세션 fixture는 동기 함수라 asyncio.run으로 한 번씩 돌린다. 마이그레이션 주소는 alembic env.py가
    # config.attributes['database_url']에서 읽는다.
    asyncio.run(manage_database(True))
    try:
        cfg = Config(str(BACKEND / "alembic.ini"))
        cfg.attributes["database_url"] = test_url
        command.upgrade(cfg, "head")
        yield test_url
    finally:
        asyncio.run(manage_database(False))


# 테스트마다 테스트 DB 엔진을 만든다. 끝나면 users 행을 모두 지운다. translations는 users에 ON DELETE
# CASCADE로 묶여 함께 지워지므로 다음 테스트가 빈 DB에서 시작한다(DB를 매번 새로 만드는 것보다 빠름).
@pytest.fixture
async def db_engine(migrated_database):
    engine = create_async_engine(migrated_database, **ENGINE_OPTIONS)
    try:
        yield engine
    finally:
        async with engine.begin() as connection:
            await connection.execute(delete(User))
        await engine.dispose()


# 테스트 코드가 DB를 직접 읽고 쓸 때 쓰는 세션. expire_on_commit=False라 커밋한 뒤에도 객체 속성을 다시 읽지
# 않고 쓸 수 있다.
@pytest.fixture
async def db_session(db_engine):
    async with async_sessionmaker(db_engine, expire_on_commit=False)() as session:
        yield session


# API를 부르는 비동기 HTTP 클라이언트. 앱의 get_db 의존성을 테스트 DB 세션으로 바꿔 끼우고, 끝나면 원래
# 덮어쓰기 목록을 되돌려 다른 테스트에 영향이 남지 않게 한다. ASGITransport라 실제 포트를 열지 않는다.
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


# bcrypt 해시는 일부러 느리므로 테스트 비밀번호의 해시를 세션에서 한 번만 만든다.
@pytest.fixture(scope="session")
def password_hash():
    return hash_password("password123")


# 테스트 DB에 저장한 사용자 한 명(person@example.com / password123).
@pytest.fixture
async def user(db_session, password_hash):
    account = User(email="person@example.com", hashed_password=password_hash)
    db_session.add(account)
    await db_session.commit()
    return account


# 그 사용자의 접근 토큰을 담은 Authorization 헤더. 로그인 과정 없이 인증이 필요한 API를 부를 때 쓴다.
@pytest.fixture
def auth_headers(user):
    return {"Authorization": f"Bearer {create_token(user.id, 'access')}"}

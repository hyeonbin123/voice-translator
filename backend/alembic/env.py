# [파일 역할] Alembic 마이그레이션 실행 환경이다. `uv run alembic upgrade head`(README 로컬 개발 2단계)나
# 테스트가 Alembic 명령을 부르면 이 파일이 실행되어 alembic/versions/의 마이그레이션을 DB에 적용한다.
# [설계 이유] 앱은 asyncpg(비동기 드라이버)로 DB에 붙으므로 같은 DATABASE_URL로 비동기 엔진을 만들고, 동기
# API인 Alembic 마이그레이션은 connection.run_sync 안에서 돌린다. 이렇게 하면 마이그레이션용 동기 드라이버를
# 따로 설치하지 않아도 된다.
# [흐름] 파일 맨 아래에서 세 경우를 고른다: (1) 오프라인 모드면 SQL만 만든다 (2) 호출한 쪽이 이미 연 연결을
# 넘겼으면 그 연결로 바로 적용한다 (3) 아니면 새 비동기 엔진을 만들어 적용한다.
# [연결] tests/conftest.py는 config.attributes["database_url"]에 임의 이름의 테스트 DB 주소를,
# tests/test_migrations.py는 config.attributes["connection"]에 열린 연결을 넣어 이 파일을 쓴다.

import asyncio

# pool.NullPool: 마이그레이션은 한 번 쓰고 끝나므로 연결을 풀에 남기지 않는다. create_async_engine: 앱과 같은
# asyncpg 주소를 그대로 쓰기 위한 비동기 엔진.
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import create_async_engine

# Alembic의 실행 문맥, 앱 설정(기본 DB 주소), 모든 ORM 모델이 등록된 Base.metadata를 가져온다.
# app.models import는 이름을 쓰지 않아도 모델 클래스를 정의시켜 Base.metadata에 테이블을 등록하려는 것이다.
# 그래야 autogenerate·check가 모델과 마이그레이션을 비교할 수 있다.
from alembic import context
from app.config import get_settings
from app.db.base import Base
from app.models import AudioFile, Translation, User  # noqa: F401

# config: alembic.ini와 호출한 쪽이 넘긴 attributes를 담은 설정 객체. target_metadata: 모델 정의(비교 기준).
config = context.config
target_metadata = Base.metadata


# 마이그레이션할 DB 주소를 고른다. 호출한 쪽이 attributes로 주소를 넘겼으면(테스트 DB) 그것을, 아니면 앱
# 설정의 DATABASE_URL을 쓴다. 테스트가 개발 DB를 건드리지 않게 하는 장치다.
def database_url():
    return config.attributes.get("database_url", get_settings().database_url)


# 오프라인 모드(`alembic upgrade --sql`): DB에 붙지 않고 실행할 SQL 문만 만든다. literal_binds=True는 바인드
# 변수 대신 값을 SQL에 직접 써 넣어 사람이 읽을 수 있는 스크립트로 만든다.
def run_migrations_offline():
    context.configure(url=database_url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


# 열린 동기 연결 하나로 마이그레이션을 실제 적용한다. 비동기 엔진 경로(run_sync)와 테스트가 넘긴 연결 경로가
# 함께 쓴다.
# compare_type=True: `alembic check`·autogenerate가 열 이름뿐 아니라 열 타입(String 길이 등)의 차이도 잡게
# 한다. tests/test_migrations.py가 모델과 마이그레이션이 같은지 이것으로 확인한다.
def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


# 기본 경로: 비동기 엔진으로 연결을 열고, 동기 함수 do_run_migrations를 run_sync로 그 연결 위에서 실행한다.
# 끝나면 성공·실패와 상관없이 엔진을 정리한다.
async def run_async_migrations():
    engine = create_async_engine(database_url(), poolclass=pool.NullPool)
    try:
        async with engine.connect() as connection:
            await connection.run_sync(do_run_migrations)
    finally:
        await engine.dispose()


# 실행 경로 선택. 테스트가 연결을 넘긴 경우(test_migrations.py)는 이미 이벤트 루프 안의 run_sync에서 불리므로
# asyncio.run을 다시 부르면 안 된다. 그래서 그 연결로 바로 적용하고, 그 밖의 경우에만 새 이벤트 루프를 연다.
if context.is_offline_mode():
    run_migrations_offline()
elif connection := config.attributes.get("connection"):
    do_run_migrations(connection)
else:
    asyncio.run(run_async_migrations())

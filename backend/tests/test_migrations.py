# [파일 역할] Alembic 마이그레이션이 SQLAlchemy 모델과 일치하고, 처음 상태(base)까지 내렸다가 다시 올려도
# 깨지지 않는지 확인한다.
# [왜 필요한가] 스키마를 바꿀 때 모델과 마이그레이션을 함께 고치는 것이 규칙이다. 한쪽만 고치면
# command.check가 '새 마이그레이션이 필요하다'고 실패해 CI에서 잡힌다.

from alembic.config import Config
from sqlalchemy import inspect

# alembic.command: check(모델과 DB 스키마 차이 확인)·upgrade·downgrade를 코드에서 부른다.
from alembic import command


# 순서: 차이 없음 확인 → 세 테이블 존재 → base로 내려 세 테이블이 모두 사라짐 → head로 다시 올림 → 다시 차이
# 없음 확인. downgrade가 테이블을 남기거나 upgrade를 다시 할 수 없는 마이그레이션을 막는다.
# Alembic 명령은 동기 API라, 비동기 엔진의 연결을 run_sync로 넘기고 cfg.attributes["connection"]에 담아
# alembic/env.py가 새 연결을 만들지 않고 이 연결을 쓰게 한다.
async def test_migration_matches_models_and_roundtrips(db_engine):
    def check(connection):
        cfg = Config("alembic.ini")
        cfg.attributes["connection"] = connection
        command.check(cfg)
        assert {"users", "translations", "audio_files"} <= set(inspect(connection).get_table_names())
        command.downgrade(cfg, "base")
        assert not {"users", "translations", "audio_files"} & set(inspect(connection).get_table_names())
        command.upgrade(cfg, "head")
        command.check(cfg)

    # This engine only ever points at the randomly named test database from conftest.
    async with db_engine.begin() as connection:
        await connection.run_sync(check)

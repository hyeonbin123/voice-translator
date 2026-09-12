from alembic.config import Config
from sqlalchemy import inspect

from alembic import command


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

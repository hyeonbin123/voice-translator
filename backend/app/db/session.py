from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import get_settings

# hide_parameters: without it a database error's message lists the bound values, which include the
# user's text and its translation, and would reach the log (T54). Tests build their engine with these too.
ENGINE_OPTIONS = {"pool_pre_ping": True, "hide_parameters": True}
engine = create_async_engine(get_settings().database_url, **ENGINE_OPTIONS)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

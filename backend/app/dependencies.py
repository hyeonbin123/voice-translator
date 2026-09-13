from collections.abc import AsyncIterator
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.security import TokenKind, decode_token
from app.db.session import SessionLocal
from app.models import User
from app.services.audio_store import AudioStore
from app.services.pipeline import PipelineModels

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


async def get_db() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session


DbSession = Annotated[AsyncSession, Depends(get_db)]


def unauthorized() -> HTTPException:
    return HTTPException(401, "Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})


async def user_from_token(token: str, kind: TokenKind, db: AsyncSession) -> User:
    try:
        user_id = decode_token(token, kind)
    except jwt.InvalidTokenError as exc:
        raise unauthorized() from exc
    user = await db.get(User, user_id)
    if user is None:
        raise unauthorized()
    return user


async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)], db: DbSession) -> User:
    return await user_from_token(token, "access", db)


CurrentUser = Annotated[User, Depends(get_current_user)]


def get_models(request: Request) -> PipelineModels:
    # T11 assigns PipelineModels to app.state.models during lifespan startup.
    return getattr(request.app.state, "models", PipelineModels(stt=None, translator=None))


def get_audio_store() -> AudioStore:
    return AudioStore(get_settings().audio_dir)


Models = Annotated[PipelineModels, Depends(get_models)]
StoredAudio = Annotated[AudioStore, Depends(get_audio_store)]

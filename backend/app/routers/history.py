from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Response
from sqlalchemy import delete, func, select
from sqlalchemy.orm import selectinload

from app.dependencies import CurrentUser, DbSession, StoredAudio
from app.models import AudioFile, Translation
from app.schemas.history import HistoryItem, HistoryPage

router = APIRouter(prefix="/history", tags=["history"])


@router.get("", response_model=HistoryPage)
async def list_history(
    user: CurrentUser,
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> HistoryPage:
    owned = Translation.user_id == user.id
    total = await db.scalar(select(func.count()).select_from(Translation).where(owned))
    items = await db.scalars(
        select(Translation)
        .where(owned)
        .options(selectinload(Translation.audio_file))
        .order_by(Translation.created_at.desc(), Translation.id.desc())
        .limit(limit)
        .offset(offset)
    )
    return HistoryPage(items=[HistoryItem.model_validate(item) for item in items], total=total or 0)


@router.get("/{id}", response_model=HistoryItem)
async def get_history(id: UUID, user: CurrentUser, db: DbSession) -> Translation:
    item = await db.scalar(
        select(Translation)
        .where(Translation.id == id, Translation.user_id == user.id)
        .options(selectinload(Translation.audio_file))
    )
    if item is None:
        raise HTTPException(404, "History not found")
    return item


@router.delete("/{id}", status_code=204)
async def delete_history(id: UUID, user: CurrentUser, db: DbSession, store: StoredAudio) -> Response:
    path = await db.scalar(
        select(AudioFile.path).join(Translation).where(Translation.id == id, Translation.user_id == user.id)
    )
    deleted_id = await db.scalar(
        delete(Translation)
        .where(Translation.id == id, Translation.user_id == user.id)
        .returning(Translation.id)
    )
    if deleted_id is None:
        raise HTTPException(404, "History not found")
    await db.commit()
    # PostgreSQL cascades audio_files rows. Disk failures must not undo the DB deletion.
    if path is not None:
        await store.delete(path)
    return Response(status_code=204)

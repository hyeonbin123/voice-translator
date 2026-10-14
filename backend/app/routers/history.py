# [파일 역할] 내 번역 기록 API: 목록(GET /api/history), 하나 보기(GET /api/history/{id}), 삭제(DELETE
# /api/history/{id}).
# [설계 이유] 모든 쿼리에 'user_id = 지금 사용자' 조건을 넣어 본인 기록만 다룬다. 남의 기록과 없는 기록은 같은
# 404로 답해 다른 사용자의 기록이 있는지 알리지 않는다(docs/api.md 번역 기록). 기록을 만드는 API는 따로 없고
# 번역 API가 성공한 번역을 저장한다(services/pipeline.save).
# [연결] 응답 형태는 schemas/history.py, 음성 파일 삭제는 services/audio_store.py가 맡는다.

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Response
from sqlalchemy import delete, func, select
from sqlalchemy.orm import selectinload

from app.dependencies import CurrentUser, DbSession, StoredAudio
from app.models import AudioFile, Translation
from app.schemas.history import HistoryItem, HistoryPage

router = APIRouter(prefix="/history", tags=["history"])


# 페이지 단위 목록. limit 1~100(기본 20), offset 0 이상을 Query로 검증해 범위를 벗어나면 422.
# total은 페이지와 상관없는 본인 기록 전체 개수라 화면이 페이지 수를 계산할 수 있다. 정렬은 created_at
# 내림차순, 시각이 같으면 id 내림차순으로 해 페이지를 넘겨도 순서가 흔들리지 않는다. selectinload로 음성 행을
# 한 번에 미리 불러온다(모델의 lazy="raise" 때문에 필요, audio_id 계산에 씀).
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


# 기록 하나. id가 UUID 형식이 아니면 FastAPI가 422, 본인 기록이 아니면 404.
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


# 기록 삭제. 순서: (1) 지우기 전에 음성 파일 경로를 읽어 둔다 (2) 본인 기록만 지우는 DELETE ... RETURNING으로
# 실제로 지웠는지 확인한다(없으면 404) (3) commit하면 audio_files 행은 DB의 ON DELETE CASCADE로 함께 지워진다
# (4) 마지막에 디스크 파일을 지운다.
# 파일을 DB commit 뒤에 지우는 이유: 파일 삭제가 실패해도 DB 삭제는 되돌리지 않고 로그만 남긴다(docs/api.md
# 저장과 삭제). 반대 순서면 DB 오류 때 기록은 남고 음성만 사라질 수 있다.
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
    # audio_files 행은 PostgreSQL이 CASCADE로 함께 지웠다. 디스크 삭제가 실패해도 DB 삭제를 되돌리지 않는다.
    if path is not None:
        await store.delete(path)
    return Response(status_code=204)

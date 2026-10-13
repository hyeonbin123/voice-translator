# [파일 역할] GET /api/audio/{id}: 번역 음성(WAV)을 로그인한 본인에게만 내려준다.
# [설계 이유] 화면의 <audio src>는 인증 헤더를 붙일 수 없어서, 화면이 이 경로를 fetch로 받아 blob URL로
# 재생한다(docs/api.md). 음성은 개인 기록이므로 공개 정적 파일로 두지 않고 인증된 요청으로만 준다(README 설계
# 판단).
# [연결] DB의 audio_files 행에서 경로를 찾고, services/audio_store.AudioStore가 디스크에서 읽는다.

from uuid import UUID

from fastapi import APIRouter, HTTPException, Response
from sqlalchemy import select

from app.dependencies import CurrentUser, DbSession, StoredAudio
from app.models import AudioFile, Translation

router = APIRouter(prefix="/audio", tags=["audio"])


# 음성 ID와 '그 음성이 딸린 번역의 주인 = 지금 사용자'를 한 쿼리(조인)로 함께 확인한다. 남의 음성, 없는 ID,
# 기록 삭제로 지워진 파일 모두 같은 404를 내 다른 사용자의 음성이 있는지조차 알리지 않는다.
# Cache-Control: private, no-store — 공유 캐시(프록시)나 브라우저 디스크 캐시에 개인 음성이 남지 않게 한다.
@router.get("/{id}")
async def get_audio(id: UUID, user: CurrentUser, db: DbSession, store: StoredAudio) -> Response:
    path = await db.scalar(
        select(AudioFile.path).join(Translation).where(AudioFile.id == id, Translation.user_id == user.id)
    )
    content = await store.read(path) if path is not None else None
    if content is None:
        raise HTTPException(404, "Audio not found")
    return Response(content, media_type="audio/wav", headers={"Cache-Control": "private, no-store"})

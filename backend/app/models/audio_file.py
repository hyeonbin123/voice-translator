# [파일 역할] audio_files 테이블의 ORM 모델. 번역 한 건에 딸린 합성 음성 파일(번역 음성) 하나의 메타데이터다.
# [설계 이유] 음성 자체는 DB가 아니라 디스크(AUDIO_DIR, services/audio_store.py)에 두고 여기에는 파일 이름과
# 길이만 둔다. 녹음 원본은 저장하지 않으므로(README 설계 판단) kind는 'tts' 하나뿐이다. 스키마는
# alembic/versions/0001과 같아야 한다.
# [연결] services/pipeline.save가 만들고, routers/audio.py가 파일 경로를 찾을 때, routers/history.py가 삭제
# 전에 경로를 읽을 때 쓴다.

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

# 타입 검사 때만 Translation을 가져온다. 실행 중에 서로 import하면 순환 import가 되므로, relationship은 문자열
# "Translation"으로 가리킨다.
if TYPE_CHECKING:
    from app.models.translation import Translation


# 검사 제약: kind는 'tts'만, 길이(ms)는 음수가 아님. 제약 이름은 naming convention으로 ck_audio_files_kind
# 등이 된다.
class AudioFile(Base):
    __tablename__ = "audio_files"
    __table_args__ = (
        CheckConstraint("kind = 'tts'", name="kind"),
        CheckConstraint("duration_ms >= 0", name="duration"),
    )

    # id는 API의 audio_id(GET /api/audio/{id})로 그대로 쓰인다. UUID라 추측해서 남의 음성 주소를 맞히기
    # 어렵고, 소유자 확인은 routers/audio.py가 따로 한다.
    # translation_id: unique=True로 번역당 음성 하나. 번역이 지워지면 DB가 이 행도 지운다(ON DELETE CASCADE).
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    translation_id: Mapped[UUID] = mapped_column(
        ForeignKey("translations.id", ondelete="CASCADE"), unique=True
    )
    # default는 파이썬에서 행을 만들 때, server_default는 DB 쪽 기본값이다.
    # 마이그레이션(server_default="tts")과 맞춰 `alembic check`가 차이를 보지 않게 둘 다 준다.
    # path: AUDIO_DIR 안의 상대 파일 이름("<uuid>.wav"). 응답에는 노출하지 않는다.
    kind: Mapped[str] = mapped_column(String(16), default="tts", server_default="tts")
    path: Mapped[str] = mapped_column(Text)
    duration_ms: Mapped[int]
    # lazy="raise": 비동기 세션에서는 속성에 처음 접근할 때 몰래 쿼리를 보내는 지연 로딩이
    # 오류(MissingGreenlet)로 이어진다. 그래서 미리 불러오지 않은 채 접근하면 바로 예외를 내게 해 실수를
    # 테스트에서 드러낸다.
    translation: Mapped["Translation"] = relationship(back_populates="audio_file", lazy="raise")

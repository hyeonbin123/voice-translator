# [파일 역할] translations 테이블의 ORM 모델. 성공한 번역 한 건(기록 한 줄)이다.
# [설계 이유] 원문·번역문과 함께 단계별 모델 이름(stt/mt/tts_model)과 걸린 시간(ms)을 남겨, 기록 화면에서 어떤
# 모델이 얼마나 걸렸는지 보여 준다(docs/plan.md 5장). 녹음 원본은 저장하지 않고 인식된 글자만 source_text로
# 남긴다.
# [연결] services/pipeline.save가 만들고, routers/history.py가 조회·삭제하며, schemas/history.HistoryItem이 이
# 객체의 속성을 그대로 읽어(from_attributes) 응답을 만든다. 스키마는 alembic/versions/0001과 같아야 한다.

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

# 순환 import를 피하려고 AudioFile은 타입 검사 때만 가져오고 relationship은 문자열로 가리킨다.
if TYPE_CHECKING:
    from app.models.audio_file import AudioFile


# 검사 제약은 API 검증과 별개로 DB에서도 허용 값(mode, 언어 쌍, 음수가 아닌 시간)을 강제한다. NULL인
# stt_ms·tts_ms는 SQL 규칙상 검사를 통과한다.
# 복합 인덱스(user_id, created_at, id): 기록 목록이 사용자로 거르고 created_at·id 내림차순으로 정렬하는
# 쿼리(routers/history.list_history)에 맞춘 것이다.
class Translation(Base):
    __tablename__ = "translations"
    __table_args__ = (
        CheckConstraint("mode IN ('text', 'speech')", name="mode"),
        CheckConstraint(
            "source_lang IN ('en', 'ko') AND target_lang IN ('en', 'ko') AND source_lang <> target_lang",
            name="language_pair",
        ),
        CheckConstraint("stt_ms >= 0 AND mt_ms >= 0 AND tts_ms >= 0", name="timings"),
        Index("ix_translations_user_created_id", "user_id", "created_at", "id"),
    )

    # user_id: 사용자가 지워지면 기록도 DB가 함께 지운다(ON DELETE CASCADE).
    # stt_*·tts_*는 Optional: 글자 입력은 인식을 하지 않고, 합성은 실패하거나 꺼져 있을 수 있다. mt_*는 번역이
    # 늘 있으므로 필수다. mt_model에는 오타 교정을 거쳤으면 "... + ollama/..."처럼 교정 모델이
    # 붙는다(services/pipeline.prepare).
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    mode: Mapped[str] = mapped_column(String(6))
    source_lang: Mapped[str] = mapped_column(String(2))
    target_lang: Mapped[str] = mapped_column(String(2))
    source_text: Mapped[str] = mapped_column(Text)
    translated_text: Mapped[str] = mapped_column(Text)
    stt_model: Mapped[str | None] = mapped_column(String(255))
    mt_model: Mapped[str] = mapped_column(String(255))
    tts_model: Mapped[str | None] = mapped_column(String(255))
    stt_ms: Mapped[int | None]
    mt_ms: Mapped[int]
    tts_ms: Mapped[int | None]
    # 시간대가 있는 시각을 DB 시계(now())로 채운다. PostgreSQL의 now()는 트랜잭션이 시작된 시각이고,
    # dependencies.get_current_user가 사용자 조회 뒤 commit하므로 HTTP 기록의 created_at은 인증 시각이 아니라
    # 저장 트랜잭션의 시각이 된다.
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # 번역 음성(없을 수 있음). cascade="all, delete-orphan": 번역을 ORM으로 저장·삭제하면 음성 행도 함께.
    # passive_deletes=True: 삭제는 DB의 ON DELETE CASCADE에 맡겨 ORM이 자식 행을 미리 읽어 오지 않게 한다.
    # lazy="raise": 비동기 세션에서 숨은 지연 로딩을 막으므로, 조회하는 쪽이 selectinload로 미리
    # 불러온다(routers/history.py).
    audio_file: Mapped["AudioFile | None"] = relationship(
        back_populates="translation", cascade="all, delete-orphan", passive_deletes=True, lazy="raise"
    )

    # API 응답의 audio_id 필드. HistoryItem이 from_attributes로 이 속성을 읽는다. 음성이 없으면 None이고, 그때
    # 화면은 브라우저 내장 음성으로 읽는다(docs/api.md).
    @property
    def audio_id(self) -> UUID | None:
        return self.audio_file.id if self.audio_file else None

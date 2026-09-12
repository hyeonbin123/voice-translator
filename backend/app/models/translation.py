from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.audio_file import AudioFile


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
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    audio_file: Mapped["AudioFile | None"] = relationship(
        back_populates="translation", cascade="all, delete-orphan", passive_deletes=True, lazy="raise"
    )

    @property
    def audio_id(self) -> UUID | None:
        return self.audio_file.id if self.audio_file else None

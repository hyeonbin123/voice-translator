from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.translation import Translation


class AudioFile(Base):
    __tablename__ = "audio_files"
    __table_args__ = (
        CheckConstraint("kind = 'tts'", name="kind"),
        CheckConstraint("duration_ms >= 0", name="duration"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    translation_id: Mapped[UUID] = mapped_column(
        ForeignKey("translations.id", ondelete="CASCADE"), unique=True
    )
    kind: Mapped[str] = mapped_column(String(16), default="tts", server_default="tts")
    path: Mapped[str] = mapped_column(Text)
    duration_ms: Mapped[int]
    translation: Mapped["Translation"] = relationship(back_populates="audio_file", lazy="raise")

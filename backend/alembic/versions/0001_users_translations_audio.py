"""Create users, translations and synthesized audio metadata."""

import sqlalchemy as sa

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("hashed_password", sa.String(60), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )
    op.create_table(
        "translations",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("mode", sa.String(6), nullable=False),
        sa.Column("source_lang", sa.String(2), nullable=False),
        sa.Column("target_lang", sa.String(2), nullable=False),
        sa.Column("source_text", sa.Text(), nullable=False),
        sa.Column("translated_text", sa.Text(), nullable=False),
        sa.Column("stt_model", sa.String(255), nullable=True),
        sa.Column("mt_model", sa.String(255), nullable=False),
        sa.Column("tts_model", sa.String(255), nullable=True),
        sa.Column("stt_ms", sa.Integer(), nullable=True),
        sa.Column("mt_ms", sa.Integer(), nullable=False),
        sa.Column("tts_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("mode IN ('text', 'speech')", name="mode"),
        sa.CheckConstraint(
            "source_lang IN ('en', 'ko') AND target_lang IN ('en', 'ko') AND source_lang <> target_lang",
            name="language_pair",
        ),
        sa.CheckConstraint("stt_ms >= 0 AND mt_ms >= 0 AND tts_ms >= 0", name="timings"),
    )
    op.create_index("ix_translations_user_created_id", "translations", ["user_id", "created_at", "id"])
    op.create_table(
        "audio_files",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "translation_id", sa.Uuid(), sa.ForeignKey("translations.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("kind", sa.String(16), server_default="tts", nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.UniqueConstraint("translation_id", name="uq_audio_files_translation_id"),
        sa.CheckConstraint("kind = 'tts'", name="kind"),
        sa.CheckConstraint("duration_ms >= 0", name="duration"),
    )


def downgrade() -> None:
    op.drop_table("audio_files")
    op.drop_index("ix_translations_user_created_id", table_name="translations")
    op.drop_table("translations")
    op.drop_table("users")

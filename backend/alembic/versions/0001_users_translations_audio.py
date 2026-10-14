"""사용자(users), 번역 기록(translations), 합성 음성 메타데이터(audio_files) 테이블을 만든다."""

# [파일 역할] 첫 (그리고 지금까지 유일한) DB 스키마 마이그레이션이다. 사용자(users), 번역 기록(translations),
# 합성 음성 파일 메타데이터(audio_files) 세 테이블을 만든다.
# [설계 이유] 스키마는 app/models/의 ORM 모델과 같아야 하고, tests/test_migrations.py가 `alembic check`로 둘이
# 같은지와 downgrade→upgrade 왕복을 확인한다. 녹음 원본은 저장하지 않으므로(docs/plan.md 5장) 음성 테이블에는
# 합성 결과(kind='tts')만 들어간다.
# [연결] alembic/env.py가 이 파일을 실행한다.

import sqlalchemy as sa

from alembic import op

# Alembic 이력 정보. down_revision=None은 이것이 첫 마이그레이션이라는 뜻이다.
revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


# 세 테이블과 기록 목록용 인덱스를 만든다. 외래 키 순서(users → translations → audio_files)대로 만든다.
def upgrade() -> None:
    # users: 이메일은 소문자로 저장해 대소문자만 다른 중복도 UNIQUE 제약으로 막는다(docs/api.md 가입). 320자는
    # 이메일 주소의 최대 길이, hashed_password 60자는 bcrypt 해시 문자열의 길이다. created_at은 DB
    # 시계(now())로 채운다.
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("hashed_password", sa.String(60), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )
    # translations: 성공한 번역 한 건이 한 행이다. 사용자를 지우면 기록도 함께 지워지도록 ON DELETE CASCADE.
    # mode는 'text'/'speech'(6자), 언어는 'en'/'ko'(2자). 모델명과 단계별 시간(ms)을 함께 남겨 어떤 모델로
    # 얼마나 걸렸는지 기록에서 볼 수 있게 한다. 글자 입력은 인식(STT)을 하지 않고 합성(TTS)은 실패할 수
    # 있으므로 stt_*·tts_*는 NULL을 허용하고, 번역(mt_*)은 늘 있으므로 NOT NULL이다.
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
        # API 검증과 별개로 DB에서도 허용 값, '출발·도착 언어가 다름', '시간은 음수가 아님'을 강제한다. 제약
        # 이름은 alembic/env.py가 넘긴 target_metadata의 naming convention(app/db/base.py)에 따라
        # `ck_translations_mode`처럼 완성된다.
        sa.CheckConstraint("mode IN ('text', 'speech')", name="mode"),
        sa.CheckConstraint(
            "source_lang IN ('en', 'ko') AND target_lang IN ('en', 'ko') AND source_lang <> target_lang",
            name="language_pair",
        ),
        sa.CheckConstraint("stt_ms >= 0 AND mt_ms >= 0 AND tts_ms >= 0", name="timings"),
    )
    # 기록 목록 조회(GET /api/history)는 user_id로 거르고 created_at·id 내림차순으로 정렬한다. 같은 열 순서의
    # 복합 인덱스로 거르기와 정렬을 함께 처리한다.
    op.create_index("ix_translations_user_created_id", "translations", ["user_id", "created_at", "id"])
    # audio_files: 번역 한 건의 합성 음성 파일 한 개(번역당 최대 하나, UNIQUE). 파일 자체는
    # 디스크(AUDIO_DIR)에 두고 여기에는 경로와 길이만 둔다. 경로는 API 응답에 노출하지 않는다(docs/api.md).
    # 번역 행이 지워지면 함께 지워진다(CASCADE).
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


# upgrade의 역순으로 지운다. 외래 키가 가리키는 쪽(users)을 마지막에 지워야 제약 오류가 나지 않는다.
def downgrade() -> None:
    op.drop_table("audio_files")
    op.drop_index("ix_translations_user_created_id", table_name="translations")
    op.drop_table("translations")
    op.drop_table("users")

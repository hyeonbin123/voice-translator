# [파일 역할] users 테이블의 ORM 모델(가입한 사용자).
# [연결] routers/auth.py가 가입 때 만들고 로그인 때 이메일로 찾으며, dependencies.user_from_token이 토큰의
# 사용자 ID로 읽는다. 사용자를 지우면 DB의 ON DELETE CASCADE로 번역 기록과 음성 행도 지워진다. 스키마는
# alembic/versions/0001과 같아야 한다.

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class User(Base):
    __tablename__ = "users"

    # id는 JWT의 sub 클레임에 들어간다. email은 소문자로 정규화해 저장하므로(schemas/auth.py) unique 제약이
    # 대소문자만 다른 중복도 막는다. 길이 320은 이메일 주소의 최대 길이다.
    # hashed_password: bcrypt 해시(60자). 비밀번호 원문은 저장하지 않는다.
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True)
    hashed_password: Mapped[str] = mapped_column(String(60))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

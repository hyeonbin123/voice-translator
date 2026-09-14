from datetime import UTC, datetime, timedelta
from typing import Literal
from uuid import UUID, uuid4

import bcrypt
import jwt

from app.config import get_settings

TokenKind = Literal["access", "refresh"]


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, hashed_password: str) -> bool:
    try:
        encoded = password.encode("utf-8")
        if len(encoded) > 72:
            return False
        return bcrypt.checkpw(encoded, hashed_password.encode("ascii"))
    except (ValueError, UnicodeError):
        return False


def create_token(user_id: UUID, kind: TokenKind) -> str:
    settings = get_settings()
    lifetime = (
        timedelta(minutes=settings.access_token_expire_minutes)
        if kind == "access"
        else timedelta(days=settings.refresh_token_expire_days)
    )
    now = datetime.now(UTC)
    return jwt.encode(
        {"sub": str(user_id), "type": kind, "iat": now, "exp": now + lifetime, "jti": str(uuid4())},
        settings.jwt_secret_key.get_secret_value(),
        algorithm="HS256",
    )


def decode_token(token: str, kind: TokenKind) -> UUID:
    return decode_token_with_expiry(token, kind)[0]


def decode_token_with_expiry(token: str, kind: TokenKind) -> tuple[UUID, datetime]:
    """The user, and when the token stops being valid: a WebSocket outlives the token it opened with."""
    payload = jwt.decode(
        token,
        get_settings().jwt_secret_key.get_secret_value(),
        algorithms=["HS256"],
        options={"require": ["sub", "type", "iat", "exp", "jti"]},
    )
    if payload["type"] != kind:
        raise jwt.InvalidTokenError("Invalid token type")
    try:
        return UUID(payload["sub"]), datetime.fromtimestamp(payload["exp"], UTC)
    except (ValueError, TypeError, AttributeError, OverflowError) as exc:
        raise jwt.InvalidTokenError("Invalid subject") from exc

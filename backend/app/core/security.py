# [파일 역할] 비밀번호 해시·검증과 JWT(access·refresh) 발급·검증을 맡는 보안 도우미 모음이다.
# [설계 이유] 인증 규칙(bcrypt 72바이트 제한, HS256 서명, 필수 클레임, access/refresh 구분)을 한 파일에 모아
# HTTP 라우터와 동시통역 웹소켓이 같은 규칙을 쓰게 한다. 규칙의 근거는 docs/api.md '인증' 절이다.
# [연결] routers/auth.py가 가입·로그인·갱신에서 hash_password·verify_password·create_token을,
# dependencies.py가 보호된 HTTP 경로에서 decode_token을, routers/live.py가 웹소켓 첫 메시지의 토큰 검증에
# decode_token_with_expiry를 쓴다.

from datetime import UTC, datetime, timedelta
from typing import Literal
from uuid import UUID, uuid4

# bcrypt: 비밀번호를 느린 해시로 저장해 DB가 새도 원래 비밀번호를 알아내기 어렵게 한다. jwt(PyJWT): 서명된
# 토큰을 만들고 서명·만료·클레임을 검증한다.
import bcrypt
import jwt

from app.config import get_settings

# 토큰 종류. 클레임 "type"에 넣어 access와 refresh를 서로 대신 쓰지 못하게 한다(docs/api.md).
TokenKind = Literal["access", "refresh"]


# 비밀번호를 bcrypt로 해시한다. gensalt()가 매번 새 salt를 만들어 같은 비밀번호도 다른 해시가 된다. 결과는
# 60자 ASCII 문자열이라 users.hashed_password(String(60))에 그대로 들어간다.
# 72바이트 초과 비밀번호는 가입 요청 검증(schemas/auth.py)에서 먼저 거절된다. bcrypt는 72바이트 뒤를
# 무시하거나(구버전) 오류를 내기 때문이다. 해시는 CPU를 오래 쓰므로 routers/auth.py가 run_in_threadpool로
# 부른다.
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")


# 로그인 비밀번호가 저장된 해시와 맞는지 확인한다. 실패 사유와 상관없이 False만 돌려줘 호출한 쪽이 같은 401을
# 내게 한다.
# 72바이트를 넘으면 bcrypt에 넘기지 않고 바로 False(docs/api.md: 로그인에서 72바이트 초과는 401). 해시
# 문자열이 깨졌거나 인코딩이 안 되면 bcrypt가 ValueError를 내므로 그것도 False로 바꾼다.
def verify_password(password: str, hashed_password: str) -> bool:
    try:
        encoded = password.encode("utf-8")
        if len(encoded) > 72:
            return False
        return bcrypt.checkpw(encoded, hashed_password.encode("ascii"))
    except (ValueError, UnicodeError):
        return False


# 사용자 ID와 종류로 HS256 JWT를 만든다. 수명은 설정값(access 분, refresh 일)을 따른다.
# 클레임: sub(사용자 ID), type(access/refresh), iat(발급 시각), exp(만료 시각), jti(토큰마다 다른 UUID). jti
# 덕분에 같은 초에 두 번 발급해도 토큰 문자열이 달라진다. 서버에 폐기 목록은 없어 이전 토큰도 만료 시각까지
# 유효하다(docs/api.md).
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


# HTTP 보호 경로용: 토큰을 검증하고 사용자 ID만 돌려준다. 실패하면 jwt.InvalidTokenError(호출한
# dependencies.user_from_token이 401로 바꿈).
def decode_token(token: str, kind: TokenKind) -> UUID:
    return decode_token_with_expiry(token, kind)[0]


def decode_token_with_expiry(token: str, kind: TokenKind) -> tuple[UUID, datetime]:
    """The user, and when the token stops being valid: a WebSocket outlives the token it opened with."""
    # 서명·만료(exp)를 검증하고, 다섯 클레임이 모두 있어야 통과시킨다(require). algorithms를 HS256 하나로
    # 고정해 토큰 머리의 알고리즘 값(예: none)으로 검증을 우회하지 못하게 한다.
    payload = jwt.decode(
        token,
        get_settings().jwt_secret_key.get_secret_value(),
        algorithms=["HS256"],
        options={"require": ["sub", "type", "iat", "exp", "jti"]},
    )
    # refresh 토큰으로 보호 경로에 들어오거나 access 토큰으로 갱신하는 것을 막는다.
    if payload["type"] != kind:
        raise jwt.InvalidTokenError("Invalid token type")
    # sub가 UUID 문자열이 아니거나 exp가 범위를 벗어나는 값이면, 서명이 맞더라도 같은 InvalidTokenError로 바꿔
    # 호출한 쪽이 한 종류의 예외만 처리하게 한다. 만료 시각을 함께 돌려주는 것은 웹소켓이 연결을 연 토큰보다
    # 오래 살 수 있어서다(routers/live.py가 만료 뒤 첫 메시지에서 연결을 닫음).
    try:
        return UUID(payload["sub"]), datetime.fromtimestamp(payload["exp"], UTC)
    except (ValueError, TypeError, AttributeError, OverflowError) as exc:
        raise jwt.InvalidTokenError("Invalid subject") from exc

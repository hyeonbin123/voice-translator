# [파일 역할] 인증 API: 가입(POST /api/auth/register), 로그인(POST /api/auth/login), 토큰 갱신(POST
# /api/auth/refresh), 내 정보(GET /api/auth/me).
# [설계 이유] 로그인 실패는 이메일이 없든 비밀번호가 틀리든 같은 401로 답하고, 없는 사용자도 bcrypt 검증을
# 똑같이 거치게 해 응답 시간으로 계정 존재를 짐작하지 못하게 한다. 비밀번호 해시·검증은 CPU를 오래 쓰므로
# 스레드 풀에서 돌린다. 토큰 응답은 캐시에 남지 않게 한다(docs/api.md 인증).
# [연결] 토큰 발급·비밀번호 처리는 core/security.py, 요청·응답 형태는 schemas/auth.py, 토큰 검증은
# dependencies.py를 쓴다.

from typing import Annotated

# run_in_threadpool: bcrypt 연산을 이벤트 루프 밖에서. OAuth2PasswordRequestForm: 로그인을 OAuth2 password
# 폼(username·password, x-www-form-urlencoded)으로 받아 Swagger UI의 Authorize 버튼과 호환되게 한다.
# IntegrityError: 이메일 중복을 DB 유일 제약 위반으로 알아낸다.
from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import EmailStr, TypeAdapter, ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.config import get_settings
from app.core.security import create_token, hash_password, verify_password
from app.dependencies import CurrentUser, DbSession, unauthorized, user_from_token
from app.models import User
from app.schemas.auth import RefreshRequest, RegisterRequest, TokenResponse, UserResponse

# email_adapter: 로그인 폼의 username은 그냥 문자열이라 이메일 형식 검증을 따로 한다.
router = APIRouter(prefix="/auth", tags=["auth"])
email_adapter = TypeAdapter(EmailStr)
# dummy_hash: 없는 사용자일 때 대신 비교할 유효한 bcrypt 해시(어떤 비밀번호와도 맞지 않음). 유효한 해시라서
# 없는 사용자 확인도 실제 사용자와 같은 비싼 bcrypt 검증 경로를 거친다.
dummy_hash = "$2b$12$C6UzMDM.H6dfI/f/IKcEe.5Z1QHj4FPpRqqlf9gHqVvvIx6aWS7/."


# access·refresh 토큰 쌍을 새로 만들어 돌려준다(로그인과 갱신이 함께 씀). Cache-Control: no-store와 HTTP/1.0
# 호환용 no-cache 헤더로 브라우저·프록시가 토큰이 든 응답을 저장하지 않게 한다. expires_in은 access 토큰
# 수명(초).
def token_pair(user: User, response: Response) -> TokenResponse:
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    return TokenResponse(
        access_token=create_token(user.id, "access"),
        refresh_token=create_token(user.id, "refresh"),
        expires_in=get_settings().access_token_expire_minutes * 60,
    )


# 가입. RegisterRequest가 이메일 형식·소문자화, 비밀번호 8자 이상·72바이트 이하를 먼저 검증한다(실패는 422).
# 중복 확인을 미리 SELECT로 하지 않고 INSERT 뒤 유일 제약 위반으로 판단한다. 두 가입 요청이 동시에 와도 DB
# 제약이 하나만 통과시키므로 경쟁 조건이 없다. 응답은 UserResponse라 해시는 나가지 않는다.
@router.post("/register", response_model=UserResponse, status_code=201)
async def register(body: RegisterRequest, db: DbSession) -> User:
    user = User(email=body.email, hashed_password=await run_in_threadpool(hash_password, body.password))
    db.add(user)
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        # 예상되는 가입 충돌은 이메일 유일 제약 위반뿐이다. 23505는 PostgreSQL의 unique_violation 코드다. 그
        # 밖의 무결성 오류는 예상하지 못한 것이라 그대로 올려 500이 되게 한다.
        if getattr(exc.orig, "sqlstate", None) == "23505":
            raise HTTPException(409, "Email already registered") from exc
        raise
    # DB가 채운 created_at(server_default)을 읽어 와 응답에 넣는다.
    await db.refresh(user)
    return user


# 로그인. username에 이메일을 받는다. 이메일 형식이 틀려도 422가 아니라 401로 답해(docs/api.md) 다른 로그인
# 실패와 구분되지 않게 한다. 저장할 때처럼 소문자로 바꿔 찾는다.
@router.post("/login", response_model=TokenResponse)
async def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()], db: DbSession, response: Response
) -> TokenResponse:
    try:
        email = str(email_adapter.validate_python(form.username)).lower()
    except ValidationError as exc:
        raise unauthorized() from exc
    user = await db.scalar(select(User).where(User.email == email))
    # 사용자가 없어도 dummy_hash로 같은 bcrypt 검증을 돌린다. 그래야 '없는 이메일'이 '틀린 비밀번호'보다 눈에
    # 띄게 빨리 실패하지 않는다(시간 차로 가입 여부를 알아내는 것을 막음). 다만 가입(register)이 이미 있는
    # 이메일에 409로 답해 같은 정보를 드러내므로, 이 방어는 로그인 응답 시간으로 알아내는 길만 막는다. 가입
    # 응답을 같게 하려면 이메일 확인 흐름이 필요해 받아들인 위험으로 둔다(docs/security.md).
    valid = await run_in_threadpool(
        verify_password, form.password, user.hashed_password if user else dummy_hash
    )
    if not valid or user is None:
        raise unauthorized()
    return token_pair(user, response)


# 토큰 갱신. access 헤더 없이 refresh 토큰을 JSON 본문으로 받는다(1~4096자). refresh 종류의 토큰만 받고,
# 성공하면 새 쌍을 발급한다. 서버에 폐기 목록이 없어 이전 refresh 토큰도 원래 만료까지 유효하다(docs/api.md).
@router.post("/refresh", response_model=TokenResponse)
async def refresh(body: RefreshRequest, db: DbSession, response: Response) -> TokenResponse:
    user = await user_from_token(body.refresh_token, "refresh", db)
    return token_pair(user, response)


# access 토큰의 사용자 정보. 화면(frontend/src/api/client.ts)은 로그인 직후와 새로고침 뒤 저장된 refresh
# 토큰이 있을 때 이 경로로 사용자 정보를 받는다.
@router.get("/me", response_model=UserResponse)
async def me(user: CurrentUser) -> User:
    return user

# [파일 역할] FastAPI 의존성 주입(Depends)용 함수와 Annotated 별칭 모음이다. 라우터는 DbSession, Sessions,
# CurrentUser, Models, StoredAudio를 매개변수 타입으로 적기만 하면 DB 세션·로그인 사용자·모델 묶음·음성
# 저장소를 받는다.
# [설계 이유] 이렇게 한곳에서 주입하면 테스트가 app.dependency_overrides[get_models] 등으로 실제
# 모델·DB·디스크를 가짜로 바꿔 끼울 수 있다(tests/test_translate_api.py, test_live_api.py 등).
# [연결] routers/audio·auth·health·history·live·translate가 이 별칭을 쓴다.

from collections.abc import AsyncIterator
from typing import Annotated

# jwt: 토큰 검증 실패 예외(InvalidTokenError)를 잡으려고. HTTPConnection: HTTP 요청과 웹소켓의 공통 부모라 둘
# 다에서 app.state를 읽을 수 있다. OAuth2PasswordBearer: Authorization: Bearer 헤더에서 토큰을 꺼내고 Swagger
# UI에 로그인 버튼을 붙인다.
import jwt
from fastapi import Depends, HTTPException
from fastapi.requests import HTTPConnection
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import get_settings
from app.core.security import TokenKind, decode_token
from app.db.session import SessionLocal
from app.models import User
from app.services.audio_store import AudioStore
from app.services.pipeline import PipelineModels

# 헤더가 없거나 Bearer가 아니면 FastAPI가 401 "Not authenticated"를 낸다(docs/api.md 오류 응답). tokenUrl은
# Swagger 문서에 로그인 경로를 알려 주는 용도다.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


# 요청 하나에 세션 하나를 열고, 응답이 끝나면 async with가 세션을 닫아 연결을 풀에 돌려준다.
async def get_db() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session


DbSession = Annotated[AsyncSession, Depends(get_db)]


# 세션 대신 세션 팩토리를 주입한다. routers/live.py가 기록을 저장할 때만 짧게 세션을 연다. 테스트는 이 함수를
# 테스트 DB 팩토리로 바꿔 끼운다.
def get_sessions() -> async_sessionmaker[AsyncSession]:
    """필요할 때만 세션을 여는 핸들러를 위한 세션 팩토리를 돌려준다.

    동시통역 웹소켓은 사람이 말하는 동안 계속 열려 있으므로, 그 시간 내내 DB 연결을 붙잡고 있으면 안 된다.
    """
    return SessionLocal


Sessions = Annotated[async_sessionmaker[AsyncSession], Depends(get_sessions)]


# 인증 실패를 모두 같은 401 본문으로 만든다. 실패 이유(서명·만료·없는 사용자)를 나누지 않아 계정 존재 여부를
# 알리지 않는다(docs/api.md 오류 응답). WWW-Authenticate: Bearer 헤더는 HTTP 401의 표준 형식이다.
def unauthorized() -> HTTPException:
    return HTTPException(401, "Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})


# 토큰을 검증하고(종류 kind 포함) 그 사용자를 DB에서 읽는다. 토큰이 맞아도 사용자가 지워졌으면 401. 보호된
# HTTP 경로(get_current_user, access)와 토큰 갱신(routers/auth.refresh, refresh)이 함께 쓴다.
async def user_from_token(token: str, kind: TokenKind, db: AsyncSession) -> User:
    try:
        user_id = decode_token(token, kind)
    except jwt.InvalidTokenError as exc:
        raise unauthorized() from exc
    user = await db.get(User, user_id)
    if user is None:
        raise unauthorized()
    return user


# 보호된 경로의 로그인 사용자. access 토큰만 받는다.
async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)], db: DbSession) -> User:
    user = await user_from_token(token, "access", db)
    # 사용자 조회 트랜잭션을 여기서 끝낸다. 그래야 번역이 모델 스레드를 기다리고 그 위에서 도는 동안 풀의
    # 연결을 붙잡지 않는다(docs/api.md: 다른 요청이 막히지 않음). expire_on_commit=False라 `user`의 값은
    # 그대로 남고, 핸들러의 다음 쿼리(pipeline.save)는 새롭고 짧은 트랜잭션을 연다.
    await db.commit()
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


# app/main.py의 lifespan이 올려 둔 모델 묶음을 꺼낸다. LOAD_MODELS=false(테스트·CI)라 app.state.models가
# 없으면 인식·번역이 None인 빈 묶음을 돌려주고, 번역 경로는 이를 보고 503으로 답한다.
def get_models(connection: HTTPConnection) -> PipelineModels:
    # lifespan 시작 단계가 app.state.models에 PipelineModels를 넣는다. Request가 아니라 HTTPConnection을
    # 받는 이유: 동시통역 웹소켓도 모델이 필요하다.
    return getattr(connection.app.state, "models", PipelineModels(stt=None, translator=None))


# 합성 음성 파일을 AUDIO_DIR에 저장·삭제하는 저장소. 테스트는 tmp_path 아래 폴더로 바꿔 끼운다.
def get_audio_store() -> AudioStore:
    return AudioStore(get_settings().audio_dir)


# 라우터 매개변수에 쓰는 별칭. 예: `models: Models`라고 적으면 get_models의 결과가 들어온다.
Models = Annotated[PipelineModels, Depends(get_models)]
StoredAudio = Annotated[AudioStore, Depends(get_audio_store)]

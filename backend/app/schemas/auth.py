# [파일 역할] 인증 API의 요청·응답 형태(pydantic 모델)와 입력 검증 규칙.
# [설계 이유] 검증을 스키마에 두면 FastAPI가 라우터 함수보다 먼저 검사해 규칙 위반을 표준 422 오류로
# 돌려준다(docs/api.md). 응답 모델을 따로 두어 비밀번호 해시가 응답에 섞이지 않게 한다.
# [연결] routers/auth.py가 쓴다.

from datetime import datetime
from typing import Literal
from uuid import UUID

# EmailStr는 이메일 형식을 검증한다(email-validator 패키지 사용).
from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


# 가입 요청 본문. 비밀번호는 8자 이상이어야 하고, max_length=72(글자 수)로 먼저 거른 뒤 아래 검증기에서 UTF-8
# 바이트 수를 다시 본다. 공백을 자르거나 정규화하지 않는다(docs/api.md).
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)

    # 이메일 전체를 소문자로 바꿔 저장·비교한다. 대소문자만 다른 이메일로 두 번 가입하는 것을 users의 유일
    # 제약이 막게 하려는 것이다.
    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return value.lower()

    # bcrypt는 72바이트까지만 쓰므로(bcrypt 5.x는 넘으면 오류) 바이트 기준으로 거절한다. 한글은 한 글자가
    # 3바이트라 `가` 24개가 한도다(docs/api.md).
    @field_validator("password")
    @classmethod
    def validate_password(cls, value: str) -> str:
        if len(value.encode("utf-8")) > 72:
            raise ValueError("Password must be at most 72 UTF-8 bytes")
        return value


# 사용자 응답(가입·/me). from_attributes=True로 ORM User 객체에서 바로 만든다. id·email·created_at만 내보낸다.
class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: EmailStr
    created_at: datetime


# 토큰 갱신 요청. 지나치게 긴 문자열로 서버에 부담을 주지 않게 길이를 1~4096자로 제한한다.
class RefreshRequest(BaseModel):
    refresh_token: str = Field(min_length=1, max_length=4096)


# 로그인·갱신 응답. token_type은 OAuth2 관례대로 "bearer", expires_in은 access 토큰 수명(초).
class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int

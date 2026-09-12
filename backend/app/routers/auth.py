from typing import Annotated

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

router = APIRouter(prefix="/auth", tags=["auth"])
email_adapter = TypeAdapter(EmailStr)
# A valid bcrypt hash keeps unknown-user checks on the same expensive code path.
dummy_hash = "$2b$12$C6UzMDM.H6dfI/f/IKcEe.5Z1QHj4FPpRqqlf9gHqVvvIx6aWS7/."


def token_pair(user: User, response: Response) -> TokenResponse:
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    return TokenResponse(
        access_token=create_token(user.id, "access"),
        refresh_token=create_token(user.id, "refresh"),
        expires_in=get_settings().access_token_expire_minutes * 60,
    )


@router.post("/register", response_model=UserResponse, status_code=201)
async def register(body: RegisterRequest, db: DbSession) -> User:
    user = User(email=body.email, hashed_password=await run_in_threadpool(hash_password, body.password))
    db.add(user)
    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        # Only the email unique constraint represents an expected registration conflict.
        if getattr(exc.orig, "sqlstate", None) == "23505":
            raise HTTPException(409, "Email already registered") from exc
        raise
    await db.refresh(user)
    return user


@router.post("/login", response_model=TokenResponse)
async def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()], db: DbSession, response: Response
) -> TokenResponse:
    try:
        email = str(email_adapter.validate_python(form.username)).lower()
    except ValidationError as exc:
        raise unauthorized() from exc
    user = await db.scalar(select(User).where(User.email == email))
    valid = await run_in_threadpool(
        verify_password, form.password, user.hashed_password if user else dummy_hash
    )
    if not valid or user is None:
        raise unauthorized()
    return token_pair(user, response)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(body: RefreshRequest, db: DbSession, response: Response) -> TokenResponse:
    user = await user_from_token(body.refresh_token, "refresh", db)
    return token_pair(user, response)


@router.get("/me", response_model=UserResponse)
async def me(user: CurrentUser) -> User:
    return user

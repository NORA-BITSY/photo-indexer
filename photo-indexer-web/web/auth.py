from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from web.config import get_settings

security = HTTPBearer(auto_error=False)


class TokenData(BaseModel):
    username: str
    exp: datetime


class User(BaseModel):
    username: str
    disabled: bool = False


def verify_password(plain_password: str, stored_password: str) -> bool:
    return plain_password == stored_password


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    settings = get_settings()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.access_token_expire_minutes
        )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, settings.secret_key, algorithm="HS256"
    )
    return encoded_jwt


def verify_token(token: str) -> Optional[TokenData]:
    try:
        settings = get_settings()
        payload = jwt.decode(token, settings.secret_key, algorithms=["HS256"])
        username: str = payload.get("sub")
        exp: int = payload.get("exp")
        if username is None or exp is None:
            return None
        return TokenData(username=username, exp=datetime.fromtimestamp(exp, tz=timezone.utc))
    except JWTError:
        return None


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> User:
    settings = get_settings()
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if credentials is None:
        raise credentials_exception

    token_data = verify_token(credentials.credentials)
    if token_data is None:
        raise credentials_exception

    # Check if token expired
    if token_data.exp < datetime.now(timezone.utc):
        raise credentials_exception

    # For simple single-user setup, just verify username matches
    if token_data.username != settings.web_username:
        raise credentials_exception

    return User(username=token_data.username)


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[User]:
    """Optional auth - returns None if no valid token."""
    if credentials is None:
        return None
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None
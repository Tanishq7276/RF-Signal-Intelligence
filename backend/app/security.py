"""Authentication and abuse protection.

* passwords are hashed with ``hashlib.scrypt`` (memory-hard, in the standard library, no extra
  dependency): stored as ``scrypt$n$r$p$salt_hex$hash_hex``
* tokens are HS256 JWTs signed with ``SIH_JWT_SECRET`` (same wire format as PyJWT, implemented here
  with hmac so the deployment has no extra dependency to audit)
* a small in-memory token bucket limits login/register and analysis submissions per client
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Any

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import get_db
from .models import User

SECRET = os.environ.get("SIH_JWT_SECRET", "dev-secret-change-me-in-production")
TOKEN_TTL_S = int(os.environ.get("SIH_JWT_TTL_S", str(24 * 3600)))
_SCRYPT = dict(n=2 ** 14, r=8, p=1, dklen=32)


# ---------------------------------------------------------------------------------------
# passwords
# ---------------------------------------------------------------------------------------
def hash_password(password: str) -> str:
    if not isinstance(password, str) or len(password) < 8:
        raise ValueError("the password must be at least 8 characters long")
    if len(password) > 256:
        raise ValueError("the password is too long (256 characters maximum)")
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode("utf-8"), salt=salt, **_SCRYPT)
    return f"scrypt${_SCRYPT['n']}${_SCRYPT['r']}${_SCRYPT['p']}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt_hex, hash_hex = str(stored).split("$")
        if scheme != "scrypt":
            return False
        dk = hashlib.scrypt(password.encode("utf-8"), salt=bytes.fromhex(salt_hex),
                            n=int(n), r=int(r), p=int(p), dklen=len(hash_hex) // 2)
        return hmac.compare_digest(dk.hex(), hash_hex)
    except Exception:
        return False


# ---------------------------------------------------------------------------------------
# JWT (HS256)
# ---------------------------------------------------------------------------------------
def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + pad)


def make_token(subject: str, claims: dict[str, Any] | None = None, ttl_s: int | None = None) -> str:
    now = int(time.time())
    payload = {"sub": subject, "iat": now, "exp": now + int(ttl_s or TOKEN_TTL_S),
               "iss": "sih26147-rf-platform", **(claims or {})}
    header = {"alg": "HS256", "typ": "JWT"}
    signing_input = f"{_b64(json.dumps(header, separators=(',', ':')).encode())}." \
                    f"{_b64(json.dumps(payload, separators=(',', ':')).encode())}"
    sig = hmac.new(SECRET.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest()
    return f"{signing_input}.{_b64(sig)}"


def decode_token(token: str) -> dict:
    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
    except ValueError as exc:
        raise ValueError("malformed token") from exc
    signing_input = f"{header_b64}.{payload_b64}"
    expected = hmac.new(SECRET.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest()
    if not hmac.compare_digest(expected, _unb64(sig_b64)):
        raise ValueError("token signature does not verify")
    payload = json.loads(_unb64(payload_b64))
    if int(payload.get("exp", 0)) < int(time.time()):
        raise ValueError("token has expired")
    return payload


# ---------------------------------------------------------------------------------------
# request dependencies
# ---------------------------------------------------------------------------------------
def bearer_token(authorization: str | None = Header(default=None)) -> str | None:
    if not authorization:
        return None
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return None


def optional_user(token: str | None = Depends(bearer_token), db: Session = Depends(get_db)) -> User | None:
    if not token:
        return None
    try:
        payload = decode_token(token)
    except ValueError:
        return None
    user = db.get(User, str(payload.get("sub")))
    return user if user and user.is_active else None


def current_user(user: User | None = Depends(optional_user)) -> User:
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="authentication required: send 'Authorization: Bearer <token>' "
                                   "(obtain one from POST /api/auth/login)",
                            headers={"WWW-Authenticate": "Bearer"})
    return user


# ---------------------------------------------------------------------------------------
# rate limiting (token bucket, per client address + bucket name)
# ---------------------------------------------------------------------------------------
_BUCKETS: dict[tuple[str, str], tuple[float, float]] = {}

LIMITS = {"auth": (10, 60.0), "upload": (60, 60.0), "analyze": (30, 60.0),
          "heavy": (20, 60.0), "default": (300, 60.0)}


def rate_limit(bucket: str = "default", request: Request | None = None, key: str | None = None) -> None:
    """Raise 429 when the client exceeds the bucket's budget. Real enforcement, per process."""
    capacity, per_seconds = LIMITS.get(bucket, LIMITS["default"])
    ident = key or (request.client.host if request and request.client else "anonymous")
    now = time.time()
    tokens, last = _BUCKETS.get((bucket, ident), (float(capacity), now))
    tokens = min(capacity, tokens + (now - last) * capacity / per_seconds)
    if tokens < 1.0:
        raise HTTPException(status_code=429, detail=f"rate limit for '{bucket}' exceeded: "
                                                    f"{capacity} requests per {per_seconds:.0f} s. "
                                                    "Wait a moment and retry.")
    _BUCKETS[(bucket, ident)] = (tokens - 1.0, now)


def find_user(db: Session, email: str) -> User | None:
    return db.scalar(select(User).where(User.email == email.strip().lower()))

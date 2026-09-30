"""Accounts: register, login, token identity.  Real JWT + scrypt password hashing."""
from __future__ import annotations

import datetime as _dt

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import AnalysisSession, UploadedFile, User
from ..security import (find_user, hash_password, make_token, optional_user, rate_limit, verify_password)
from .common import file_payload, jsonable

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=256)
    full_name: str | None = Field(default=None, max_length=120)
    organisation: str | None = Field(default=None, max_length=160)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=256)


def _workspace(db: Session, key: str) -> dict:
    files = db.scalars(select(UploadedFile).where(UploadedFile.session_key == key)).all()
    analyses = db.scalars(select(AnalysisSession).where(AnalysisSession.session_key == key)).all()
    done = [a for a in analyses if a.status == "done"]
    return {"session_key": key, "files": len(files), "analyses": len(analyses),
            "completed": len(done),
            "failed": len([a for a in analyses if a.status == "failed"])}


@router.post("/register", summary="Create an account and return a JWT")
def register(req: RegisterRequest, request: Request, db: Session = Depends(get_db)) -> dict:
    rate_limit("auth", request)
    email = req.email.strip().lower()
    if find_user(db, email):
        raise HTTPException(status_code=409, detail="an account with that email already exists")
    try:
        digest = hash_password(req.password)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    user = User(email=email, full_name=(req.full_name or None), organisation=(req.organisation or None),
                password_hash=digest, role="engineer")
    db.add(user)
    db.commit()
    db.refresh(user)
    token = make_token(user.id, {"email": user.email, "role": user.role})
    return {"token": token, "token_type": "bearer", "expires_in_s": 86400,
            "user": user.public(),
            "workspace": _workspace(db, f"u{user.id}"),
            "next": "POST /api/upload or POST /api/demo/{name} to add a signal"}


@router.post("/login", summary="Exchange email + password for a JWT")
def login(req: LoginRequest, request: Request, db: Session = Depends(get_db)) -> dict:
    rate_limit("auth", request)
    user = find_user(db, req.email)
    if user is None or not verify_password(req.password, user.password_hash or ""):
        # deliberately identical message for unknown email and wrong password
        raise HTTPException(status_code=401, detail="email or password is incorrect")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="this account is disabled")
    user.last_login_at = _dt.datetime.now(_dt.timezone.utc)
    user.login_count = int(user.login_count or 0) + 1
    db.commit()
    db.refresh(user)
    token = make_token(user.id, {"email": user.email, "role": user.role})
    return {"token": token, "token_type": "bearer", "expires_in_s": 86400, "user": user.public(),
            "workspace": _workspace(db, f"u{user.id}")}


@router.get("/me", summary="Identity of the current token")
def me(user: User = Depends(optional_user), db: Session = Depends(get_db)) -> dict:
    if user is None:
        return {"authenticated": False,
                "message": "no valid token: the workspace is being used in anonymous mode and is "
                           "isolated by the x-session-id header"}
    return {"authenticated": True, "user": user.public(), "workspace": _workspace(db, f"u{user.id}")}


@router.get("/session", summary="Workspace bound to the current identity")
def session(user: User | None = Depends(optional_user), db: Session = Depends(get_db)) -> dict:
    key = f"u{user.id}" if user else None
    return jsonable({"authenticated": bool(user), "user": user.public() if user else None,
                     "workspace": _workspace(db, key) if key else None})


@router.post("/logout", summary="Discard the token (stateless JWT)")
def logout(user: User | None = Depends(optional_user)) -> dict:
    return {"ok": True,
            "message": "the token is stateless: the client deletes it. Any token already issued stays "
                       "valid until it expires - rotate SIH_JWT_SECRET to invalidate all tokens."}

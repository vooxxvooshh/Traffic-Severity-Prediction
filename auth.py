import os
import re
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from database import get_connection, init_db

JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-on-render")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "24"))

security = HTTPBearer(auto_error=False)
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def create_access_token(user_id: int, email: str) -> str:
    payload = {
        "sub": str(user_id),
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired session.") from exc


def get_user_by_email(email: str):
    with get_connection() as conn:
        return conn.execute(
            "SELECT id, name, email, password_hash FROM users WHERE email = ?",
            (email.lower().strip(),),
        ).fetchone()


def create_user(name: str, email: str, password: str) -> dict:
    email = email.lower().strip()
    if not EMAIL_RE.match(email):
        raise HTTPException(status_code=400, detail="Invalid email format.")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")

    existing = get_user_by_email(email)
    if existing:
        raise HTTPException(status_code=409, detail="An account with this email already exists.")

    password_hash = hash_password(password)
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
            (name.strip(), email, password_hash),
        )
        conn.commit()
        user_id = cursor.lastrowid

    return {"id": user_id, "name": name.strip(), "email": email}


def authenticate_user(email: str, password: str) -> dict:
    user = get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=401, detail="Email not found.")
    if not verify_password(password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect password.")
    return {"id": user["id"], "name": user["name"], "email": user["email"]}


def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict:
    token = None
    if credentials and credentials.credentials:
        token = credentials.credentials
    elif request.cookies.get("access_token"):
        token = request.cookies.get("access_token")

    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated. Please log in.")

    payload = decode_token(token)
    return {"id": int(payload["sub"]), "email": payload["email"]}


def setup_auth() -> None:
    init_db()

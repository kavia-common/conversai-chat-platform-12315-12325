from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import httpx
import jwt
from fastapi import Depends, FastAPI, HTTPException, status, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field
from sqlalchemy import text, select, or_
from sqlalchemy.orm import Session

from src.db.config import engine, get_db
from src.db.models import Base, User, Conversation, Message
from src.schemas import (
    UserCreate,
    UserLogin,
    UserRead,
    ConversationCreate,
    ConversationRead,
    MessageCreate,
    MessageRead,
    ConversationWithMessages,
)
from src.security import hash_password, verify_password

# === Configuration via environment variables ===
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-secret-change-me")  # In production override via env
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRES_MINUTES = int(os.getenv("JWT_EXPIRES_MINUTES", "60"))
# OpenAI compatible configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# OAuth2 scheme (we use bearer token but implement our own JWT issuance)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

app = FastAPI(
    title="ConversAI Chat Backend",
    description="Backend API handling authentication, user management, chat storage, and LLM integration.",
    version="0.1.0",
    openapi_tags=[
        {"name": "health", "description": "Service health and readiness checks"},
        {"name": "database", "description": "Database connectivity and maintenance"},
        {"name": "auth", "description": "Authentication and user management"},
        {"name": "conversations", "description": "Create and manage conversations"},
        {"name": "messages", "description": "Submit and retrieve messages"},
        {"name": "llm", "description": "LLM proxy/query endpoints"},
    ],
)

# Create tables on startup if they don't already exist.
# In a production system, prefer Alembic migrations.
Base.metadata.create_all(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Helpers
def _create_access_token(subject: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=JWT_EXPIRES_MINUTES))
    to_encode = {"exp": expire, "sub": subject}
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def _decode_token(token: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


async def _get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Resolve the current user from a bearer token."""
    payload = _decode_token(token)
    sub = payload.get("sub") or {}
    user_id = sub.get("id")
    email = sub.get("email")
    if not user_id or not email:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
    user = db.get(User, user_id)
    if not user or user.email != email:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


# PUBLIC_INTERFACE
@app.get("/", tags=["health"], summary="Health Check", description="Basic service-level health check.")
def health_check():
    """Simple health check endpoint."""
    return {"message": "Healthy"}


# PUBLIC_INTERFACE
@app.get(
    "/db/health",
    tags=["database"],
    summary="Database Health",
    description="Checks if the application can connect to the database and run a simple query.",
)
def db_health(db: Session = Depends(get_db)):
    """Run a trivial SELECT 1 to verify DB connectivity."""
    db.execute(text("SELECT 1"))
    return {"database": "ok"}


# === Auth Schemas ===
class TokenResponse(BaseModel):
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field("bearer", description="Token type")


class UpdateProfileRequest(BaseModel):
    display_name: Optional[str] = Field(None, description="New display name")


# === Auth Endpoints ===
# PUBLIC_INTERFACE
@app.post(
    "/auth/signup",
    tags=["auth"],
    summary="Sign up",
    description="Create a new user account",
    response_model=UserRead,
    status_code=201,
)
def signup(payload: UserCreate, db: Session = Depends(get_db)):
    """Create a new user with hashed password."""
    existing = db.execute(select(User).where(User.email == payload.email)).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(
        email=payload.email,
        password_hash=hash_password(payload.password),
        display_name=payload.display_name,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


# PUBLIC_INTERFACE
@app.post(
    "/auth/login",
    tags=["auth"],
    summary="Login",
    description="Authenticate a user and return a JWT",
    response_model=TokenResponse,
)
def login(payload: UserLogin, db: Session = Depends(get_db)):
    """Verify credentials and return JWT access token."""
    user = db.execute(select(User).where(User.email == payload.email)).scalar_one_or_none()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = _create_access_token({"id": user.id, "email": user.email})
    return TokenResponse(access_token=token, token_type="bearer")


# PUBLIC_INTERFACE
@app.get(
    "/auth/me",
    tags=["auth"],
    summary="Get current user profile",
    description="Returns the current authenticated user's profile",
    response_model=UserRead,
)
def get_me(current_user: User = Depends(_get_current_user)):
    """Return the authenticated user's profile."""
    return current_user


# PUBLIC_INTERFACE
@app.patch(
    "/auth/me",
    tags=["auth"],
    summary="Update current user profile",
    description="Update display name for the current user",
    response_model=UserRead,
)
def update_me(
    payload: UpdateProfileRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(_get_current_user),
):
    """Update the current user's display name."""
    if payload.display_name is not None:
        current_user.display_name = payload.display_name
    db.add(current_user)
    db.commit()
    db.refresh(current_user)
    return current_user


# === Conversation Endpoints ===
# PUBLIC_INTERFACE
@app.post(
    "/conversations",
    tags=["conversations"],
    summary="Create conversation",
    description="Create a new conversation for the current user",
    response_model=ConversationRead,
    status_code=201,
)
def create_conversation(
    payload: ConversationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(_get_current_user),
):
    """Create a conversation owned by the current user."""
    convo = Conversation(title=payload.title, owner_id=current_user.id)
    db.add(convo)
    db.commit()
    db.refresh(convo)
    return convo


# PUBLIC_INTERFACE
@app.get(
    "/conversations",
    tags=["conversations"],
    summary="List conversations",
    description="List conversations owned by the current user",
    response_model=List[ConversationRead],
)
def list_conversations(
    q: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(_get_current_user),
):
    """List conversations for the current user, optionally filtering by title."""
    stmt = select(Conversation).where(Conversation.owner_id == current_user.id)
    if q:
        stmt = stmt.where(or_(Conversation.title.ilike(f"%{q}%")))
    results = db.execute(stmt.order_by(Conversation.updated_at.desc())).scalars().all()
    return results


# PUBLIC_INTERFACE
@app.get(
    "/conversations/{conversation_id}",
    tags=["conversations"],
    summary="Get conversation with messages",
    description="Retrieve a conversation and its messages (owned by current user)",
    response_model=ConversationWithMessages,
)
def get_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(_get_current_user),
):
    """Fetch conversation and nested messages."""
    convo = db.get(Conversation, conversation_id)
    if not convo or convo.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    # Force load messages via simple select to respect ordering
    msgs = db.execute(
        select(Message).where(Message.conversation_id == convo.id).order_by(Message.created_at.asc())
    ).scalars().all()
    return ConversationWithMessages(conversation=convo, messages=msgs)


# PUBLIC_INTERFACE
@app.delete(
    "/conversations/{conversation_id}",
    tags=["conversations"],
    summary="Delete conversation",
    description="Delete a conversation and its messages",
    status_code=204,
)
def delete_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(_get_current_user),
):
    """Delete a conversation owned by the current user."""
    convo = db.get(Conversation, conversation_id)
    if not convo or convo.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    db.delete(convo)
    db.commit()
    return None


# === Message Endpoints ===
# PUBLIC_INTERFACE
@app.post(
    "/messages",
    tags=["messages"],
    summary="Post message",
    description="Submit a message to a conversation",
    response_model=MessageRead,
    status_code=201,
)
def post_message(
    payload: MessageCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(_get_current_user),
):
    """Append a message to a conversation. Only owner may post."""
    convo = db.get(Conversation, payload.conversation_id)
    if not convo or convo.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    msg = Message(
        conversation_id=payload.conversation_id,
        user_id=current_user.id if payload.role == "user" else None,
        role=payload.role,
        content=payload.content,
    )
    db.add(msg)
    convo.updated_at = datetime.utcnow()
    db.add(convo)
    db.commit()
    db.refresh(msg)
    return msg


# PUBLIC_INTERFACE
@app.get(
    "/conversations/{conversation_id}/messages",
    tags=["messages"],
    summary="List messages",
    description="Get messages for a conversation",
    response_model=List[MessageRead],
)
def list_messages(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(_get_current_user),
):
    """List messages for a conversation owned by the current user."""
    convo = db.get(Conversation, conversation_id)
    if not convo or convo.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    msgs = db.execute(
        select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at.asc())
    ).scalars().all()
    return msgs


# === LLM Endpoint ===
class ChatRequest(BaseModel):
    conversation_id: int = Field(..., description="Target conversation id")
    prompt: str = Field(..., min_length=1, description="User message to send to the model")
    system_prompt: Optional[str] = Field(None, description="Optional system instruction override")
    temperature: Optional[float] = Field(0.7, ge=0, le=2, description="Sampling temperature")
    model: Optional[str] = Field(None, description="Override model name")


class ChatResponse(BaseModel):
    assistant_message: str = Field(..., description="Assistant response text")
    conversation_id: int = Field(..., description="Conversation ID")


async def _call_openai_chat(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    api_key: str,
    base_url: str,
) -> str:
    """Call OpenAI-compatible chat completions API and return assistant content."""
    # Using direct HTTP to avoid tying to a specific SDK
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"LLM provider error: {resp.text}")
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            raise HTTPException(status_code=502, detail="Malformed response from LLM provider")


# PUBLIC_INTERFACE
@app.post(
    "/llm/chat",
    tags=["llm"],
    summary="Chat with LLM",
    description="Sends the conversation context and prompt to the LLM and stores both user and assistant messages.",
    response_model=ChatResponse,
)
async def llm_chat(
    payload: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(_get_current_user),
    x_openai_api_key: Optional[str] = Header(default=None, alias="X-OpenAI-API-Key"),
):
    """
    Call the configured OpenAI-compatible endpoint with conversation context.
    The API key is taken from X-OpenAI-API-Key header if provided, otherwise OPENAI_API_KEY env var.
    """
    if not OPENAI_API_KEY and not x_openai_api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured; pass X-OpenAI-API-Key header or set env var")
    api_key = x_openai_api_key or OPENAI_API_KEY
    model = payload.model or OPENAI_MODEL

    convo = db.get(Conversation, payload.conversation_id)
    if not convo or convo.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Add the user's new message first
    user_msg = Message(
        conversation_id=convo.id, user_id=current_user.id, role="user", content=payload.prompt
    )
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)

    # Build context
    existing_msgs = db.execute(
        select(Message).where(Message.conversation_id == convo.id).order_by(Message.created_at.asc())
    ).scalars().all()

    messages: List[Dict[str, str]] = []
    if payload.system_prompt:
        messages.append({"role": "system", "content": payload.system_prompt})
    for m in existing_msgs:
        messages.append({"role": m.role, "content": m.content})

    assistant_text = await _call_openai_chat(
        messages=messages,
        model=model,
        temperature=payload.temperature or 0.7,
        api_key=api_key,
        base_url=OPENAI_API_BASE,
    )

    # Store assistant response
    asst_msg = Message(
        conversation_id=convo.id, user_id=None, role="assistant", content=assistant_text
    )
    db.add(asst_msg)
    convo.updated_at = datetime.utcnow()
    db.add(convo)
    db.commit()

    return ChatResponse(assistant_message=assistant_text, conversation_id=convo.id)

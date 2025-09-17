"""
Pydantic schemas for request/response models.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, Field, EmailStr


class UserBase(BaseModel):
    email: EmailStr = Field(..., description="Unique email address for the user")
    display_name: Optional[str] = Field(None, description="Public display name")


class UserCreate(UserBase):
    password: str = Field(..., min_length=6, description="Raw password to be hashed")


class UserLogin(BaseModel):
    email: EmailStr = Field(..., description="User email for login")
    password: str = Field(..., description="User password for login")


class UserRead(UserBase):
    id: int = Field(..., description="User ID")
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ConversationCreate(BaseModel):
    title: Optional[str] = Field(None, description="Title of the conversation")


class ConversationRead(BaseModel):
    id: int
    title: Optional[str]
    owner_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class MessageCreate(BaseModel):
    conversation_id: int = Field(..., description="Target conversation ID")
    role: str = Field("user", description="Message role (user/assistant/system)")
    content: str = Field(..., min_length=1, description="Message content")


class MessageRead(BaseModel):
    id: int
    conversation_id: int
    user_id: Optional[int]
    role: str
    content: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ConversationWithMessages(BaseModel):
    conversation: ConversationRead
    messages: List[MessageRead]

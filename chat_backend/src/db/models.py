"""
SQLAlchemy ORM models for the chat backend.

Models:
- User: Basic user with email and password hash
- Conversation: Represents a chat conversation owned by a user
- Message: Individual messages in a conversation

Includes timestamp mixin and relationship configurations.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for SQLAlchemy models."""


class TimestampMixin:
    """Adds created_at and updated_at timestamp columns."""
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class User(TimestampMixin, Base):
    """Represents an application user."""
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Relationships
    conversations: Mapped[List["Conversation"]] = relationship(
        back_populates="owner", cascade="all, delete-orphan", passive_deletes=True
    )


class Conversation(TimestampMixin, Base):
    """Represents a conversation owned by a user."""
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    owner_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )

    # Relationships
    owner: Mapped[User] = relationship(back_populates="conversations")
    messages: Mapped[List["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", passive_deletes=True, order_by="Message.created_at"
    )


class Message(TimestampMixin, Base):
    """Represents a single message within a conversation."""
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    conversation_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("conversations.id", ondelete="CASCADE"), index=True, nullable=False
    )
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), index=True, nullable=True
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="user")  # user/assistant/system
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Relationships
    conversation: Mapped[Conversation] = relationship(back_populates="messages")

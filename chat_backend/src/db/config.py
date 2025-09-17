"""
Database configuration and session management for the chat backend.

This module sets up the SQLAlchemy engine and session factory using an environment
variable DATABASE_URL if provided; otherwise, it falls back to a local SQLite file.
It exposes get_db() dependency for FastAPI routes to obtain a scoped session.
"""
from __future__ import annotations

import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Prefer DATABASE_URL when available (e.g., for PostgreSQL), otherwise use local SQLite.
# Do NOT read .env directly here; expect orchestrator to provide env vars.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./chat_app.db")

# Create SQLAlchemy engine with sensible defaults.
# For SQLite, check_same_thread must be False for multithreaded FastAPI.
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, echo=False, future=True, connect_args=connect_args)

# Session factory. autocommit=False and autoflush=False are standard for explicit control.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)


# PUBLIC_INTERFACE
def get_db() -> Generator:
    """Yield a database session for request scope and ensure proper cleanup."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

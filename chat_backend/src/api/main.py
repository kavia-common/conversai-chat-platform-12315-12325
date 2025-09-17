from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.db.config import engine, get_db
from src.db.models import Base

app = FastAPI(
    title="ConversAI Chat Backend",
    description="Backend API handling authentication, user management, chat storage, and LLM integration.",
    version="0.1.0",
    openapi_tags=[
        {"name": "health", "description": "Service health and readiness checks"},
        {"name": "database", "description": "Database connectivity and maintenance"},
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


@app.get("/", tags=["health"], summary="Health Check", description="Basic service-level health check.")
def health_check():
    """Simple health check endpoint."""
    return {"message": "Healthy"}


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

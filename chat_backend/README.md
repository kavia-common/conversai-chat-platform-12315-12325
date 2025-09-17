# Chat Backend - Database Setup

This backend uses SQLAlchemy for data access. By default it will use a local SQLite database file `chat_app.db` if no DATABASE_URL environment variable is provided.

Environment variables:
- `DATABASE_URL` (optional): SQLAlchemy connection string.
  - PostgreSQL example: `postgresql+psycopg2://USER:PASSWORD@HOST:5432/DBNAME`
  - SQLite example: `sqlite:///./chat_app.db`

Models:
- `User` (id, email, password_hash, display_name, timestamps)
- `Conversation` (id, title, owner_id, timestamps)
- `Message` (id, conversation_id, user_id, role, content, timestamps)

On startup, tables are created automatically (`Base.metadata.create_all`). In production, prefer Alembic migrations.

Health endpoints:
- `GET /` basic health check
- `GET /db/health` database connectivity check

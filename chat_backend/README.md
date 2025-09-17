# Chat Backend

This backend uses FastAPI and SQLAlchemy. By default it will use a local SQLite database file `chat_app.db` if no DATABASE_URL environment variable is provided.

Environment variables:
- `DATABASE_URL` (optional): SQLAlchemy connection string.
  - PostgreSQL example: `postgresql+psycopg2://USER:PASSWORD@HOST:5432/DBNAME`
  - SQLite example: `sqlite:///./chat_app.db`
- `JWT_SECRET_KEY` (required in production): Secret for signing JWTs.
- `JWT_ALGORITHM` (default: HS256)
- `JWT_EXPIRES_MINUTES` (default: 60)
- `OPENAI_API_KEY` (required for LLM endpoints unless supplied via header)
- `OPENAI_API_BASE` (default: https://api.openai.com/v1)
- `OPENAI_MODEL` (default: gpt-4o-mini)

Copy `.env.example` to `.env` and fill in values as needed.

Models:
- `User` (id, email, password_hash, display_name, timestamps)
- `Conversation` (id, title, owner_id, timestamps)
- `Message` (id, conversation_id, user_id, role, content, timestamps)

On startup, tables are created automatically (`Base.metadata.create_all`). In production, prefer Alembic migrations.

Health endpoints:
- `GET /` basic health check
- `GET /db/health` database connectivity check

Auth endpoints:
- `POST /auth/signup` (body: UserCreate)
- `POST /auth/login` (body: UserLogin) -> { access_token, token_type }
- `GET /auth/me` (Bearer auth)
- `PATCH /auth/me` (Bearer auth, body: {display_name})

Conversation endpoints (Bearer auth):
- `POST /conversations` (body: {title})
- `GET /conversations`
- `GET /conversations/{conversation_id}`
- `DELETE /conversations/{conversation_id}`

Message endpoints (Bearer auth):
- `POST /messages` (body: MessageCreate)
- `GET /conversations/{conversation_id}/messages`

LLM endpoints (Bearer auth):
- `POST /llm/chat` (body: {conversation_id, prompt, system_prompt?, temperature?, model?})
  - Uses `OPENAI_API_KEY` env var or `X-OpenAI-API-Key` header.

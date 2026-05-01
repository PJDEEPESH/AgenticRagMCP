"""Neon PostgreSQL connection layer with async wrappers around sync SQLAlchemy."""
import asyncio
import logging
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from backend.config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_engine = None
_SessionLocal = None


def fix_neon_url(url: str) -> str:
    """Normalize a Neon PostgreSQL URL for synchronous psycopg2."""
    url = url.strip().strip('"').strip("'")

    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql+asyncpg://"):
        url = "postgresql://" + url[len("postgresql+asyncpg://"):]

    parsed = urlparse(url)
    query_params = [
        (k, v)
        for k, v in parse_qsl(parsed.query)
        if k not in ("channel_binding", "connect_timeout")
    ]

    keys = {k for k, _ in query_params}
    if "sslmode" not in keys:
        query_params.append(("sslmode", "require"))
    query_params.append(("connect_timeout", "10"))

    new_query = urlencode(query_params)
    return urlunparse(parsed._replace(query=new_query))


class AsyncResultWrapper:
    """Wraps a sync SQLAlchemy Result so async code can call its methods uniformly."""

    def __init__(self, result):
        self._result = result

    def fetchall(self):
        return self._result.fetchall()

    def scalar(self):
        return self._result.scalar()

    def keys(self):
        return self._result.keys()


class AsyncSessionWrapper:
    """Async-friendly wrapper around a sync SQLAlchemy Session."""

    def __init__(self, session: Session):
        self._session = session

    async def execute(self, statement, params=None):
        def _run():
            if params is not None:
                return self._session.execute(statement, params)
            return self._session.execute(statement)

        result = await asyncio.to_thread(_run)
        return AsyncResultWrapper(result)

    async def commit(self):
        await asyncio.to_thread(self._session.commit)

    async def rollback(self):
        await asyncio.to_thread(self._session.rollback)

    async def close(self):
        await asyncio.to_thread(self._session.close)


def init_engine():
    """Create the SQLAlchemy engine and sessionmaker exactly once."""
    global _engine, _SessionLocal
    if _engine is not None:
        return

    fixed_url = fix_neon_url(settings.NEON_DATABASE_URL)
    _engine = create_engine(
        fixed_url,
        pool_size=2,
        max_overflow=3,
        pool_pre_ping=True,
        pool_timeout=15,
        pool_recycle=300,
    )
    _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    logger.info("Neon SQLAlchemy engine initialized.")


class neon_session:
    """Async context manager that yields an AsyncSessionWrapper for one unit of work."""

    def __init__(self):
        self._session = None
        self._wrapper = None

    async def __aenter__(self) -> AsyncSessionWrapper:
        init_engine()
        self._session = await asyncio.to_thread(_SessionLocal)
        self._wrapper = AsyncSessionWrapper(self._session)
        return self._wrapper

    async def __aexit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                await self._wrapper.commit()
            else:
                await self._wrapper.rollback()
        finally:
            await self._wrapper.close()


async def setup_vector_table():
    """Create the pgvector extension, document_chunks table, and supporting indexes."""
    async with neon_session() as session:
        await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await session.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    doc_name VARCHAR(300) NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding vector(768),
                    search_vector tsvector,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP DEFAULT NOW()
                )
                """
            )
        )

        try:
            await session.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx "
                    "ON document_chunks USING ivfflat (embedding vector_cosine_ops) "
                    "WITH (lists = 50)"
                )
            )
        except Exception as e:
            logger.warning(f"Could not create IVFFlat index: {e}")

        try:
            await session.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS document_chunks_search_idx "
                    "ON document_chunks USING GIN (search_vector)"
                )
            )
        except Exception as e:
            logger.warning(f"Could not create GIN index: {e}")

    logger.info("document_chunks table and indexes ready.")


async def check_db_health() -> bool:
    """Quick health probe used by the /health endpoint."""
    try:
        async with neon_session() as session:
            result = await session.execute(text("SELECT 1"))
            return result.scalar() == 1
    except Exception as e:
        logger.warning(f"DB health check failed: {e}")
        return False

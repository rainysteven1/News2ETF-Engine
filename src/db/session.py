"""
Database session management for PostgreSQL using SQLAlchemy.

Provides async database access with connection pooling.
"""

import os
from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.common.config import get_config
from src.db.models import Base


def get_db_url() -> str:
    """Get database URL. Env var DATABASE_URL takes precedence over config.toml."""
    env_url = os.environ.get("DATABASE_URL")
    if env_url:
        return env_url
    return get_config().database.url


# Global engine and session maker
_engine = None
_session_maker = None


def get_engine():
    """Get or create the database engine."""
    global _engine
    if _engine is None:
        url = get_db_url()
        _engine = create_engine(
            url,
            pool_pre_ping=True,  # Verify connections before using
            pool_size=5,  # Number of connections to maintain
            max_overflow=10,  # Additional connections allowed beyond pool_size
            echo=False,  # Set to True for SQL query logging
        )
    return _engine


def get_session_maker():
    """Get or create the session maker."""
    global _session_maker
    if _session_maker is None:
        engine = get_engine()
        _session_maker = sessionmaker(bind=engine, expire_on_commit=False)
    return _session_maker


def init_db():
    """Create all tables in the database."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Get a database session with automatic cleanup.

    Usage:
        with get_session() as session:
            session.query(Task).all()
    """
    session_maker = get_session_maker()
    session = session_maker()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session_sync() -> Session:
    """Get a session for manual management (caller must close).

    Usage:
        session = get_session_sync()
        try:
            ...
            session.commit()
        finally:
            session.close()
    """
    session_maker = get_session_maker()
    return session_maker()

"""
db/connection.py

SQLAlchemy engine and session management for SafeShell Member 5.

Default backend is SQLite; switch to PostgreSQL (or any other SQLAlchemy
dialect) by setting the ``SAFESHELL_DB_URL`` environment variable.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from safeshell_member5 import config

_engine: Engine | None = None


def get_engine(url: str | None = None) -> Engine:
    """Return (and lazily create) the global SQLAlchemy engine.

    Parameters
    ----------
    url : str, optional
        Override the database URL.  Passing a value replaces the cached
        engine, which is useful for tests (``sqlite:///:memory:``).
    """
    global _engine
    if url is not None or _engine is None:
        _engine = create_engine(url or config.DATABASE_URL, echo=False)
    return _engine


def get_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    """Return a ``sessionmaker`` bound to *engine*."""
    if engine is None:
        engine = get_engine()
    return sessionmaker(bind=engine)


@contextmanager
def get_session(engine: Engine | None = None) -> Generator[Session, None, None]:
    """Context manager that yields a ``Session`` with auto-commit / rollback."""
    factory = get_session_factory(engine)
    session = factory()
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()

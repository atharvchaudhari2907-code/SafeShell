"""
db/migrations.py

Idempotent schema initialisation for SafeShell Member 5.

Calling ``init_db()`` creates all ORM-defined tables if they do not already
exist.  Safe to call on every application start.
"""

from __future__ import annotations

from sqlalchemy.engine import Engine

from safeshell_member5.db.connection import get_engine
from safeshell_member5.db.models import Base


def init_db(engine: Engine | None = None) -> None:
    """Create all tables that do not yet exist.

    Parameters
    ----------
    engine : Engine, optional
        SQLAlchemy engine to use.  Defaults to the global engine returned
        by ``get_engine()``.
    """
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(engine)

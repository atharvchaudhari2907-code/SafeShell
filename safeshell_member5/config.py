"""
config.py

Centralised settings for SafeShell Member 5.

All tunable thresholds and weights are defined here so that every module
references one source of truth.  Override the database URL by setting the
SAFESHELL_DB_URL environment variable (e.g. a PostgreSQL connection string).
"""

from __future__ import annotations

import os

# ── Database ──────────────────────────────────────────────────────────
DATABASE_URL: str = os.environ.get(
    "SAFESHELL_DB_URL", "sqlite:///safeshell.db"
)

# ── Trust-score computation weights (must sum to 1.0) ─────────────────
TRUST_WEIGHT_FAMILIARITY: float = 0.30
TRUST_WEIGHT_SAFETY: float = 0.25
TRUST_WEIGHT_APPROVAL: float = 0.25
TRUST_WEIGHT_CONSISTENCY: float = 0.20

# ── Familiarity ──────────────────────────────────────────────────────
# Number of command uses at which the familiarity component saturates to 1.0.
FAMILIARITY_SATURATION: int = 50

# ── Inactivity decay ─────────────────────────────────────────────────
# After this many days of inactivity the cached trust score decays to 0.
INACTIVITY_FULL_DECAY_DAYS: float = 90.0

# ── Context-hash sliding window ──────────────────────────────────────
CONTEXT_HASH_WINDOW: int = 20

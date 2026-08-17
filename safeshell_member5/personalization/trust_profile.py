"""
personalization/trust_profile.py

User Trust Profile manager for SafeShell.

Tracks per-(user_id, command_pattern) trust state.  Each profile records
usage frequency, risk history, user accept/reject decisions, and
context-hash consistency.  A materialised ``current_trust_score`` is
recomputed on every ``record_usage()`` call and can be explicitly
refreshed via ``refresh_trust_score()``.

Cold-start policy
-----------------
When no profile exists for a (user, pattern) pair:

* ``get_profile()`` returns ``None``.
* ``get_or_create()`` creates a record with **trust_score = 0.0** —
  the command is treated as unknown and full friction applies.

Trust-score formula
-------------------
::

    familiarity  = min(usage_frequency / SATURATION, 1.0)
    safety       = 1.0 - historical_risk_avg   (0.0 when no history)
    approval     = accept / (accept + reject)   (0.0 when no decisions)
    consistency  = context_consistency_score

    base = W_fam * familiarity
         + W_saf * safety
         + W_app * approval
         + W_con * consistency

    decay = max(0, 1 - days_inactive / FULL_DECAY_DAYS)

    trust_score = base * decay          (clamped to [0, 1])

All weights and thresholds are defined in ``safeshell_member5.config``.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from safeshell_member5 import config
from safeshell_member5.db.models import TrustProfileRecord


class UserTrustProfile:
    """CRUD + scoring for per-user, per-command-pattern trust profiles."""

    def __init__(self, session: Session) -> None:
        self._session = session

    # ── Read / Create ─────────────────────────────────────────────────

    def get_or_create(
        self,
        user_id: str,
        command_pattern: str,
    ) -> TrustProfileRecord:
        """Return the existing profile or create a cold-start one.

        Cold-start defaults
        ~~~~~~~~~~~~~~~~~~~
        * ``current_trust_score = 0.0``
        * ``usage_frequency = 0``
        * all counters at zero, ``context_hashes = []``
        """
        record = self._lookup(user_id, command_pattern)
        if record is not None:
            return record

        record = TrustProfileRecord(
            user_id=user_id,
            command_pattern=command_pattern,
            usage_frequency=0,
            last_used_at=None,
            historical_risk_avg=0.0,
            accept_count=0,
            reject_count=0,
            context_consistency_score=0.0,
            current_trust_score=0.0,
            context_hashes=[],
            created_at=datetime.now(timezone.utc),
        )
        self._session.add(record)
        self._session.flush()
        return record

    def get_profile(
        self,
        user_id: str,
        command_pattern: str,
    ) -> Optional[TrustProfileRecord]:
        """Return the profile if it exists, or ``None`` (cold start)."""
        return self._lookup(user_id, command_pattern)

    # ── Record usage ──────────────────────────────────────────────────

    def record_usage(
        self,
        user_id: str,
        command_pattern: str,
        risk_score: float,
        user_decision: str,
        context_hash: str,
    ) -> TrustProfileRecord:
        """Record one command-usage event and recompute the trust score.

        Parameters
        ----------
        user_id : str
            Unique user identifier.
        command_pattern : str
            Normalised command signature (e.g. ``"rm -rf <path>"``).
        risk_score : float
            Risk score assigned to this execution (0.0–1.0).
        user_decision : str
            ``"accept"`` or ``"reject"`` — the user's response to the
            policy advisory for this command.
        context_hash : str
            Hash of the normalised execution context, used to track
            context consistency over the sliding window.
        """
        record = self.get_or_create(user_id, command_pattern)

        # ── Usage frequency & timestamp ───────────────────────────────
        record.usage_frequency += 1
        record.last_used_at = datetime.now(timezone.utc)

        # ── Running-average risk ──────────────────────────────────────
        n = record.usage_frequency
        old_avg = record.historical_risk_avg
        # Welford-style incremental mean
        record.historical_risk_avg = old_avg + (risk_score - old_avg) / n

        # ── Decision counters ─────────────────────────────────────────
        decision = user_decision.strip().lower()
        if decision in ("accept", "accepted"):
            record.accept_count += 1
        elif decision in ("reject", "rejected"):
            record.reject_count += 1

        # ── Context-hash sliding window ───────────────────────────────
        hashes: list[str] = list(record.context_hashes or [])
        hashes.append(context_hash)
        if len(hashes) > config.CONTEXT_HASH_WINDOW:
            hashes = hashes[-config.CONTEXT_HASH_WINDOW :]
        # Reassign the list so SQLAlchemy detects the mutation.
        record.context_hashes = hashes

        # ── Recompute derived scores ──────────────────────────────────
        record.context_consistency_score = self._compute_consistency(hashes)
        record.current_trust_score = self._compute_trust_score(record)

        self._session.flush()
        return record

    # ── Refresh (for time-decay recalculation) ────────────────────────

    def refresh_trust_score(
        self,
        user_id: str,
        command_pattern: str,
        *,
        now: datetime | None = None,
    ) -> Optional[TrustProfileRecord]:
        """Recompute and persist the trust score.

        Useful for applying inactivity decay without recording a new
        usage event.  Pass *now* to simulate a future point in time
        (primarily for testing).
        """
        record = self._lookup(user_id, command_pattern)
        if record is None:
            return None
        record.current_trust_score = self._compute_trust_score(
            record, now=now,
        )
        self._session.flush()
        return record

    # ── Internal helpers ──────────────────────────────────────────────

    def _lookup(
        self, user_id: str, command_pattern: str,
    ) -> Optional[TrustProfileRecord]:
        return (
            self._session.query(TrustProfileRecord)
            .filter_by(user_id=user_id, command_pattern=command_pattern)
            .first()
        )

    @staticmethod
    def _compute_consistency(hashes: list[str]) -> float:
        """Dominance ratio of the most-frequent hash in the window."""
        if not hashes:
            return 0.0
        most_common_count = Counter(hashes).most_common(1)[0][1]
        return most_common_count / len(hashes)

    @staticmethod
    def _compute_trust_score(
        record: TrustProfileRecord,
        *,
        now: datetime | None = None,
    ) -> float:
        """Derive trust score from the profile's current state.

        The score is a weighted blend of four components (familiarity,
        safety history, approval ratio, context consistency) multiplied
        by an inactivity-decay factor.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        # ── Component: familiarity ────────────────────────────────────
        familiarity = min(
            record.usage_frequency / config.FAMILIARITY_SATURATION, 1.0,
        )

        # ── Component: safety history ─────────────────────────────────
        # Only meaningful when there is actual history; otherwise zero
        # so that cold-start profiles produce trust = 0.0.
        if record.usage_frequency > 0:
            safety = 1.0 - record.historical_risk_avg
        else:
            safety = 0.0

        # ── Component: approval ratio ─────────────────────────────────
        total_decisions = record.accept_count + record.reject_count
        approval = (
            record.accept_count / total_decisions
            if total_decisions > 0
            else 0.0
        )

        # ── Component: context consistency ────────────────────────────
        consistency = record.context_consistency_score

        # ── Weighted base score ───────────────────────────────────────
        base = (
            config.TRUST_WEIGHT_FAMILIARITY * familiarity
            + config.TRUST_WEIGHT_SAFETY * safety
            + config.TRUST_WEIGHT_APPROVAL * approval
            + config.TRUST_WEIGHT_CONSISTENCY * consistency
        )

        # ── Inactivity decay ─────────────────────────────────────────
        if record.last_used_at is not None:
            last_used = record.last_used_at
            # SQLite may return timezone-naive datetimes.
            if last_used.tzinfo is None:
                last_used = last_used.replace(tzinfo=timezone.utc)
            days_inactive = (now - last_used).total_seconds() / 86400.0
            decay = max(
                0.0, 1.0 - days_inactive / config.INACTIVITY_FULL_DECAY_DAYS,
            )
        else:
            # Never used — all components are zero so decay is irrelevant.
            decay = 1.0

        return round(max(0.0, min(base * decay, 1.0)), 6)

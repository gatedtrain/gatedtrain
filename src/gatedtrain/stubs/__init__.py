"""Tier A stand-ins (skeleton-stub, CPU only; MVP-SPEC section 3).

Every module in this package is a skeleton-stub: a tiny task, a tiny
character-level transformer, a supervised trainer and a sampling evaluator.
They write the file formats of NeMo RL at 4f39fd8 and NeMo Evaluator at
ea35ba3 so the real ledger writer and pairing code read them unchanged.
"""

STUB_MARKER_FILE = "SKELETON_STUB"


def write_marker(directory, what: str) -> None:
    """Mark a stand-in output directory (skeleton-stub, MVP-SPEC sections 3 and 5)."""
    from pathlib import Path

    Path(directory, STUB_MARKER_FILE).write_text(
        f"skeleton-stub: {what}. prototype, unpromotable; not a NeMo RL or NeMo Evaluator output.\n",
        encoding="utf-8",
    )

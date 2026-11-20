"""Tier A stand-ins (skeleton-stub, CPU only; MVP-SPEC section 3).

Every module in this package is a skeleton-stub: a tiny task, a tiny
character-level transformer, a supervised trainer and a sampling evaluator.
They write the file formats of NeMo RL at 4f39fd8 and NeMo Evaluator at
ea35ba3 so the real ledger writer and pairing code read them unchanged.
"""

import os

# The stand-ins train on one thread with deterministic algorithms (model.py), which makes a run reproducible on one
# machine but not across CPUs: PyTorch picks its CPU kernels by instruction set (AVX2, AVX-512, ...) and MKL its BLAS
# code path, so the demo's gate could PASS on one runner and FAIL on another. Pin both before torch is imported (an
# operator's own setting wins): ATen's portable kernels and MKL's conditional numerical reproducibility mode.
os.environ.setdefault("ATEN_CPU_CAPABILITY", "default")
os.environ.setdefault("MKL_CBWR", "COMPATIBLE")

STUB_MARKER_FILE = "SKELETON_STUB"


def write_marker(directory, what: str) -> None:
    """Mark a stand-in output directory (skeleton-stub, MVP-SPEC sections 3 and 5)."""
    from pathlib import Path

    Path(directory, STUB_MARKER_FILE).write_text(
        f"skeleton-stub: {what}. prototype, unpromotable; not a NeMo RL or NeMo Evaluator output.\n",
        encoding="utf-8",
    )

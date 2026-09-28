"""Filesystem locations shared by every pipeline step.

Each location can be overridden with an environment variable:

    AT_DATA     raw and prepared datasets        (default: <repo>/data)
    AT_MODELS   local model checkpoints          (default: <repo>/models)
    AT_RESULTS  everything the pipeline writes   (default: <repo>/results)
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = Path(os.environ.get("AT_DATA", ROOT / "data"))
MODELS = Path(os.environ.get("AT_MODELS", ROOT / "models"))
RESULTS = Path(os.environ.get("AT_RESULTS", ROOT / "results"))

AUDIT = RESULTS / "audit"
ANALYSIS = RESULTS / "analysis"
COMPARISON = RESULTS / "comparison"
WILDCHAT = RESULTS / "wildchat"
VALIDATION = RESULTS / "validation"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"


def ensure(path: Path) -> Path:
    """Create ``path`` (a directory) if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path

"""Make ER-1 importable when pytest is launched from the repo root."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ER1_SRC = ROOT / "third_party" / "er1" / "src"
for path in (ROOT, ER1_SRC):
    text = str(path)
    if path.is_dir() and text not in sys.path:
        sys.path.insert(0, text)

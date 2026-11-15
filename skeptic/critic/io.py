"""Load ScoreSets from the open-volume layout. Validation arrays only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from skeptic.critic.baselines import G2_NAMES, REGISTERED_NAMES
from skeptic.critic.types import BaselineSet, ScoreSet


def load_scoreset(open_root, cell: str, sha: str, n_id: int, ledger_row: dict) -> ScoreSet:
    """Read ``runs/open/<cell>/<sha>/seed<k>.npy`` and split at ``n_id``."""
    folder = Path(open_root) / "runs" / "open" / cell / sha
    paths = sorted(folder.glob("seed*.npy"), key=_seed_index)
    if not paths:
        raise FileNotFoundError(folder)
    ident, oos = [], []
    for path in paths:
        array = np.load(path)
        if array.shape[0] <= n_id:
            raise ValueError(f"{path} is shorter than n_id={n_id}")
        ident.append(array[:n_id])
        oos.append(array[n_id:])
    val = ledger_row.get("val") or {}
    ece = val.get("ece")
    return ScoreSet(
        name=sha,
        arm=str(ledger_row["arm"]),
        val_id=tuple(ident),
        val_oos=tuple(oos),
        status=str(ledger_row.get("status", "ok")),
        parents=tuple(ledger_row.get("parents") or ()),
        ece=None if ece is None else tuple(ece),
        family=str(ledger_row.get("family") or ""),
    )


def load_baselines(out_dir, cell: str, n_id: int) -> BaselineSet:
    root = Path(out_dir)
    plain = json.loads((root / "plain_ece.json").read_text(encoding="utf-8"))
    scores = {}
    for name in REGISTERED_NAMES:
        folder = root / name
        if not folder.is_dir():
            continue
        paths = sorted(folder.glob("seed*.npy"), key=_seed_index)
        ident, oos = [], []
        for path in paths:
            array = np.load(path)
            ident.append(array[:n_id])
            oos.append(array[n_id:])
        arm = "edl" if name.endswith("_edl") else "ce"
        scores[name] = ScoreSet(name=name, arm=arm, val_id=tuple(ident), val_oos=tuple(oos))
    missing = [name for name in G2_NAMES if name not in scores]
    if missing:
        raise FileNotFoundError(f"baseline dump is missing {missing}")
    return BaselineSet(
        cell=cell,
        scores=scores,
        g2_names=G2_NAMES,
        plain_ece={arm: tuple(values) for arm, values in plain.items()},
    )


def _seed_index(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("seed"):
        raise ValueError(path.name)
    return int(stem.removeprefix("seed"))

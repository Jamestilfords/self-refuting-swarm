"""Score sources. Truth code never opens a sealed path itself.

``SealedSource`` asks ``skeptic.store`` to verify ``runs/DECISIONS.sha256``
before it reads. Unit tests use ``InMemorySource`` only, plus a temp directory
and a fake store for the guard.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


class GuardError(RuntimeError):
    """The sealed volume is not open, or the decisions hash does not match."""


def hash_scores(id_scores, oos_scores) -> str:
    """sha256 of the float64 bytes and shapes, in seed order."""
    import hashlib

    digest = hashlib.sha256()
    for array in list(id_scores) + list(oos_scores):
        block = np.ascontiguousarray(array, dtype=np.float64)
        digest.update(np.asarray(block.shape, dtype=np.int64).tobytes())
        digest.update(block.tobytes())
    return digest.hexdigest()


def _as_seeds(rows) -> list[np.ndarray]:
    out = []
    for row in rows:
        block = np.ascontiguousarray(np.asarray(row, dtype=np.float64))
        if block.ndim != 1 or block.size == 0 or not np.isfinite(block).all():
            raise ValueError("scores must be non-empty finite 1-D arrays")
        out.append(block)
    if len({block.shape[0] for block in out}) != 1:
        raise ValueError("score lengths differ across seeds")
    return out


@runtime_checkable
class ScoreSource(Protocol):
    kind: str

    def get(self, cell: str, name: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """``(id_per_seed, oos_per_seed)`` for a proposal sha or a baseline name."""

    def has(self, cell: str, name: str) -> bool: ...

    def labels_oos_type(self, cell: str) -> np.ndarray | None:
        """Banking77 test labels: 1 = ID-OOS, 2 = OOD-OOS."""

    def content_hash(self, cell: str, name: str) -> str: ...


class InMemorySource:
    """Synthetic or fixture scores. ``data`` maps ``(cell, name)`` to ``(ids, oos)``."""

    def __init__(
        self,
        kind: str,
        data: dict,
        oos_types: dict | None = None,
        probabilities: dict | None = None,
        plain_ece: dict | None = None,
        arms: dict | None = None,
    ):
        if kind not in ("val", "test"):
            raise ValueError("kind must be 'val' or 'test'")
        self.kind = kind
        self._data = {
            (str(cell), str(name)): (_as_seeds(ids), _as_seeds(oos))
            for (cell, name), (ids, oos) in data.items()
        }
        self._oos_types = None if oos_types is None else {str(k): np.asarray(v) for k, v in oos_types.items()}
        self._probabilities = probabilities or {}
        self._plain_ece = plain_ece or {}
        self._arms = arms or {}

    def has(self, cell: str, name: str) -> bool:
        return (str(cell), str(name)) in self._data

    def get(self, cell: str, name: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
        key = (str(cell), str(name))
        if key not in self._data:
            raise KeyError(f"no scores for {cell}/{name}")
        return self._data[key]

    def labels_oos_type(self, cell: str) -> np.ndarray | None:
        if self._oos_types is None:
            return None
        return self._oos_types.get(str(cell))

    def content_hash(self, cell: str, name: str) -> str:
        ids, oos = self.get(cell, name)
        return hash_scores(ids, oos)

    def probabilities(self, cell: str, name: str):
        return self._probabilities.get((str(cell), str(name)))

    def plain_ece(self, cell: str, arm: str) -> float | None:
        if (str(cell), str(arm)) not in self._plain_ece:
            return None
        return float(self._plain_ece[(str(cell), str(arm))])

    def arm_of(self, cell: str, name: str) -> str | None:
        return self._arms.get((str(cell), str(name)))


def _seed_paths(folder: Path, prefix: str = "seed") -> list[Path]:
    paths = []
    for path in folder.glob(f"{prefix}*.npy"):
        suffix = path.stem[len(prefix):]
        if suffix.isdigit():
            paths.append(path)
    return sorted(paths, key=lambda path: int(path.stem[len(prefix):]))


def _cell_dir(root: Path, cell: str) -> Path:
    return root.joinpath(*str(cell).split("/"))


class _FolderSource:
    def __init__(self, root, n_id_by_cell: dict):
        self.root = Path(root)
        self.n_id_by_cell = {str(cell): int(n_id) for cell, n_id in n_id_by_cell.items()}
        self._memo: dict[tuple[str, str], tuple[list[np.ndarray], list[np.ndarray]]] = {}

    def _folder(self, cell: str, name: str) -> Path:
        return _cell_dir(self.root, cell) / str(name)

    def has(self, cell: str, name: str) -> bool:
        folder = self._folder(cell, name)
        return folder.is_dir() and bool(_seed_paths(folder))

    def _read(self, cell: str, name: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if cell not in self.n_id_by_cell:
            raise KeyError(f"no n_id registered for {cell}")
        folder = self._folder(cell, name)
        paths = _seed_paths(folder)
        if not paths:
            raise FileNotFoundError(f"no seed arrays in {folder}")
        n_id = self.n_id_by_cell[cell]
        ids: list[np.ndarray] = []
        oos: list[np.ndarray] = []
        for path in paths:
            array = np.asarray(np.load(path), dtype=np.float64)
            if array.ndim != 1 or array.shape[0] <= n_id:
                raise ValueError(f"{path} is shorter than n_id={n_id}")
            ids.append(np.ascontiguousarray(array[:n_id]))
            oos.append(np.ascontiguousarray(array[n_id:]))
        return ids, oos

    def get(self, cell: str, name: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
        key = (str(cell), str(name))
        if key not in self._memo:
            self._memo[key] = self._read(cell, name)
        self._after_read(cell, name)
        return self._memo[key]

    def _after_read(self, cell: str, name: str) -> None:
        return None

    def labels_oos_type(self, cell: str) -> np.ndarray | None:
        path = _cell_dir(self.root, cell) / "oos_type.npy"
        if not path.is_file():
            return None
        return np.asarray(np.load(path))

    def content_hash(self, cell: str, name: str) -> str:
        ids, oos = self.get(cell, name)
        return hash_scores(ids, oos)

    def probabilities(self, cell: str, name: str):
        folder = self._folder(cell, name)
        paths = _seed_paths(folder, prefix="probs_seed")
        if not paths:
            return None
        n_id = self.n_id_by_cell[cell]
        rows = []
        for path in paths:
            labels_path = path.with_name("labels_" + path.name[len("probs_"):])
            if not labels_path.is_file():
                return None
            probs = np.asarray(np.load(path), dtype=np.float64)
            labels = np.asarray(np.load(labels_path))
            if len(probs) != n_id:
                probs = probs[:n_id]
            if len(labels) != len(probs):
                labels = labels[:len(probs)]
            rows.append((probs, labels))
        return rows

    def plain_ece(self, cell: str, arm: str) -> float | None:
        path = _cell_dir(self.root, cell) / "plain_ece.json"
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload.get(arm)
        if not values:
            return None
        return float(np.mean(np.asarray(values, dtype=np.float64)))

    def arm_of(self, cell: str, name: str) -> str | None:
        meta_path = self._folder(cell, name) / "meta.json"
        if meta_path.is_file():
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            if payload.get("arm"):
                return str(payload["arm"])
        if str(name).endswith("_edl"):
            return "edl"
        if str(name).endswith("_ce"):
            return "ce"
        return None


class OpenSource(_FolderSource):
    """Validation scores under ``open_root/<cell>/<name>/seed<k>.npy``.

    Arrays are canonical ``val`` then ``oos_val`` and are split at ``n_id``.
    Baseline directories written by ``python -m skeptic.critic.baselines`` use
    the same layout when ``--out`` is the cell directory.
    """

    kind = "val"

    def __init__(self, open_root, n_id_by_cell: dict):
        super().__init__(open_root, n_id_by_cell)
        self.kind = "val"


def _repo_of(sealed_root: Path) -> Path:
    if sealed_root.name == "sealed" and sealed_root.parent.name == "runs":
        return sealed_root.parent.parent
    return sealed_root.parent


def _require_unsealed(repo: Path | None = None):
    """Return the published-freeze permit. Construction fails closed without it."""
    import skeptic.store as store

    root = Path(repo) if repo is not None else Path(__file__).resolve().parents[2]
    try:
        return store.verify_published_freeze(root)
    except store.GuardError as exc:
        raise GuardError(str(exc)) from exc


class SealedSource(_FolderSource):
    """Test scores. The constructor refuses unless the decisions hash matches.

    Every ``get`` appends one line to ``<sealed_root>/../guard.log``.
    """

    kind = "test"

    def __init__(self, sealed_root, n_id_by_cell: dict, repo: Path | None = None):
        _require_unsealed(repo if repo is not None else _repo_of(Path(sealed_root)))
        super().__init__(sealed_root, n_id_by_cell)
        self.kind = "test"
        self.guard_log = self.root.parent / "guard.log"

    def _after_read(self, cell: str, name: str) -> None:
        self.guard_log.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).isoformat()
        line = f'{stamp}, purpose="truth", cell="{cell}", name="{name}"\n'
        with self.guard_log.open("a", encoding="utf-8") as handle:
            handle.write(line)


class SealedNpzSource:
    """Sealed ``scores/<group>/<sha>.npz`` files plus ``plain_ece.json``.

    Names in ``references[cell]`` resolve to those hashes. A 12-hex name is
    the file hash. The constructor requires a published freeze, and every
    array's ``decision_commit`` and ``decision_digest`` must match the permit.
    """

    kind = "test"

    def __init__(self, sealed_root, permit, references: dict, arms: dict | None = None,
                 repo: Path | None = None):
        checked = _require_unsealed(repo if repo is not None else _repo_of(Path(sealed_root)))
        if checked.commit != permit.commit or checked.digest != permit.digest:
            raise GuardError("Sealed permit does not match the published freeze")
        self.root = Path(sealed_root)
        self.permit = permit
        self.references = {str(cell): {str(name): str(sha) for name, sha in mapping.items()}
                           for cell, mapping in references.items()}
        self._arms = {((str(cell), str(name)) if isinstance(key, tuple) else key): value
                      for key, value in (arms or {}).items()}
        self.guard_log = self.root / "guard.log"
        self._memo: dict[tuple[str, str], tuple[list[np.ndarray], list[np.ndarray]]] = {}
        self.kind = "test"

    def _sha(self, cell: str, name: str) -> str:
        named = self.references.get(str(cell), {})
        if name in named:
            return named[name]
        if len(str(name)) == 12 and all(ch in "0123456789abcdef" for ch in str(name)):
            return str(name)
        raise KeyError(f"{cell} has no sealed scores for {name}")

    def _path(self, cell: str, sha: str) -> Path:
        from skeptic.score_artifacts import CELLS

        group = CELLS[cell]
        return self.root / "scores" / group / f"{sha}.npz"

    def has(self, cell: str, name: str) -> bool:
        try:
            path = self._path(cell, self._sha(cell, name))
        except KeyError:
            return False
        return path.is_file() and not path.is_symlink()

    def _load(self, cell: str, name: str):
        sha = self._sha(cell, name)
        key = (str(cell), sha)
        if key in self._memo:
            return sha, self._memo[key]
        path = self._path(cell, sha)
        if path.is_symlink() or path.parent.is_symlink() or path.parent.parent.is_symlink():
            raise GuardError(f"Sealed score path is a symlink: {path}")
        if not path.is_file():
            raise GuardError(f"Sealed score is missing: {path}")
        from skeptic.rules.common import TEST_N_IN

        with np.load(path, allow_pickle=False) as saved:
            scores = np.array(saved["scores"], dtype=np.float64, copy=True)
            n_in = int(saved["n_in"])
            commit = str(saved["decision_commit"])
            digest = str(saved["decision_digest"])
            if str(saved["proposal_sha"]) != sha:
                raise GuardError(f"Sealed score identity does not match {sha}")
        if commit != self.permit.commit or digest != self.permit.digest:
            raise GuardError(f"Sealed score provenance does not match the freeze: {sha}")
        if scores.ndim != 2 or scores.shape[0] != 5 or not np.isfinite(scores).all():
            raise GuardError(f"Sealed scores are not five finite seeds: {sha}")
        if n_in != TEST_N_IN[cell] or not 0 < n_in < scores.shape[1]:
            raise GuardError(f"Sealed n_in for {cell} is {n_in}, expected {TEST_N_IN[cell]}")
        ident = [np.ascontiguousarray(row[:n_in]) for row in scores]
        oos = [np.ascontiguousarray(row[n_in:]) for row in scores]
        self._memo[key] = (ident, oos)
        return sha, self._memo[key]

    def get(self, cell: str, name: str):
        _sha, pair = self._load(cell, name)
        self._after_read(cell, name)
        return pair

    def content_hash(self, cell: str, name: str) -> str:
        ident, oos = self._load(cell, name)[1]
        return hash_scores(ident, oos)

    def probabilities(self, cell: str, name: str):
        return None

    def labels_oos_type(self, cell: str):
        return None

    def arm_of(self, cell: str, name: str) -> str | None:
        from skeptic.rules.common import REFERENCE_ARMS

        direct = self._arms.get((str(cell), str(name)))
        if direct:
            return str(direct)
        if str(name) in REFERENCE_ARMS:
            return REFERENCE_ARMS[str(name)]
        if str(name).endswith("_edl"):
            return "edl"
        if str(name).endswith("_ce"):
            return "ce"
        return None

    def plain_ece(self, cell: str, arm: str) -> float | None:
        path = self.root / "plain_ece.json"
        if path.is_symlink() or not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("decision_commit") != self.permit.commit \
                or payload.get("decision_digest") != self.permit.digest:
            raise GuardError("plain_ece.json provenance does not match the freeze")
        values = (payload.get(cell) or {}).get(arm)
        if not values:
            return None
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1 or array.size != 5 or not np.isfinite(array).all():
            raise GuardError(f"plain-model ECE for {cell}/{arm} is not five finite seeds")
        return float(array.mean())

    def _after_read(self, cell: str, name: str) -> None:
        self.guard_log.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).isoformat()
        line = f'{stamp}, purpose="truth", cell="{cell}", name="{name}"\n'
        with self.guard_log.open("a", encoding="utf-8") as handle:
            handle.write(line)

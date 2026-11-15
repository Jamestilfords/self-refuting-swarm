"""Bridge: feed the scored stream (runs/ledger.jsonl + per-item score files) to the critic in skeptic.critic.

    cell = load_cell("clinc150/bert")                 # scores from runs/scores, downloaded from the open volume
    decision = verdict(cell.candidates[sha], cell.incumbent, cell.baselines, LordPlusPlus(), cell.cell)

    python -m skeptic.critic_inputs                   # smoke: the four canaries against CE energy, every cell

Score files: <scores>/<group>/<sha>.npz with scores (seeds, n) and n_in; the first n_in items are in-scope.
The critic imports ER-1 directly, so third_party/er1/src must be on PYTHONPATH and torch must be installed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GROUPS = {"clinc150/bert": "clinc150_bert", "clinc150/roberta": "clinc150_roberta",
          "banking77_oos/bert": "banking77_bert", "banking77_oos/roberta": "banking77_roberta"}
START = "energy_ce"                      # starting incumbent and recall reference
GUARD_STATUS = "io"                      # the validator stopped a file or network access: the lock, verdict BUG


@dataclass
class Cell:
    cell: str
    candidates: dict       # sha -> ScoreSet, for every proposal the critic can judge (status ok or guard)
    baselines: object      # BaselineSet
    incumbent: object      # ScoreSet of CE energy
    streams: dict          # sha -> stream tag (templated, llm, canary, baseline)
    malformed: dict        # sha -> ledger status, for proposals that never reach the critic


def _split(path: Path):
    z = np.load(path)
    n = int(z["n_in"])
    scores = np.asarray(z["scores"], np.float64)
    return tuple(s[:n] for s in scores), tuple(s[n:] for s in scores), "conf" in z.files


def load_cell(cell: str, scores_dir="runs/scores", ledger_path="runs/ledger.jsonl",
              baselines_path="proposals/baselines.json") -> Cell:
    from skeptic.critic.baselines import G2_NAMES, REGISTERED_NAMES
    from skeptic.critic.types import BaselineSet, ScoreSet

    folder = Path(scores_dir) / GROUPS[cell]
    rows = [json.loads(line) for line in Path(ledger_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = {r["sha"]: r for r in rows if r["cell"] == cell}
    named = json.loads(Path(baselines_path).read_text(encoding="utf-8"))
    named = {r["name"].replace("@", "_"): r["sha"] for r in (named["proposals"] if isinstance(named, dict) else named)}

    base, plain = {}, {}
    for name in REGISTERED_NAMES:
        row = rows.get(named.get(name, ""))
        if row is None or row["status"] != "ok":
            raise FileNotFoundError(f"{cell}: registered baseline {name} is not scored")
        val_id, val_oos, _ = _split(folder / f"{row['sha']}.npz")
        base[name] = ScoreSet(name=name, arm=row["arm"], val_id=val_id, val_oos=val_oos)
        if row["val"].get("ece_plain") is not None:
            plain[row["arm"]] = tuple(row["val"]["ece_plain"])
    if set(plain) != {"ce", "edl"}:
        raise ValueError(f"{cell}: the ledger has no plain-model ECE; rerun modal_jobs.score_stream so calibration is filled in")
    baselines = BaselineSet(cell=cell, scores=base, g2_names=G2_NAMES, plain_ece=plain)

    candidates, malformed = {}, {}
    for sha, row in rows.items():
        if row["status"] == GUARD_STATUS:
            candidates[sha] = ScoreSet(name=sha, arm=row["arm"] or "ce", val_id=(), val_oos=(), status="guard")
        elif row["status"] != "ok":
            malformed[sha] = row["status"]
        else:
            val_id, val_oos, has_probs = _split(folder / f"{sha}.npz")
            ece = row["val"].get("ece")
            if has_probs and ece is None:
                raise ValueError(f"{cell}: {sha} returns probs but has no validation ECE; rerun modal_jobs.score_stream")
            candidates[sha] = ScoreSet(name=sha, arm=row["arm"], val_id=val_id, val_oos=val_oos,
                                       parents=tuple(row.get("parents") or ()), ece=ece, family=row.get("family") or "")
    return Cell(cell, candidates, baselines, base[START], {s: r["stream"] for s, r in rows.items()}, malformed)


def main(argv=None) -> int:
    """Smoke test: the four planted canaries against CE energy in every cell."""
    import argparse
    from skeptic.critic import LordPlusPlus, verdict

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--scores", default="runs/scores")
    ap.add_argument("--ledger", default="runs/ledger.jsonl")
    ap.add_argument("--proposals", default="proposals")
    args = ap.parse_args(argv)
    canaries = json.loads((Path(args.proposals) / "canaries.json").read_text(encoding="utf-8"))
    canaries = canaries["proposals"] if isinstance(canaries, dict) else canaries
    bad = 0
    for cell_name in GROUPS:
        cell = load_cell(cell_name, args.scores, args.ledger, Path(args.proposals) / "baselines.json")
        best = max(cell.baselines.g2_names, key=lambda n: cell.baselines.scores[n].mean_auroc())
        print(f"\n{cell_name}: {len(cell.candidates)} proposals, CE energy {cell.incumbent.mean_auroc():.4f}, "
              f"strongest baseline {best} {cell.baselines.scores[best].mean_auroc():.4f}")
        for c in canaries:
            got = verdict(cell.candidates[c["sha"]], cell.incumbent, cell.baselines, LordPlusPlus(), cell_name).verdict.value
            bad += got != c["expected"]
            print(f"  canary {c['name']:<16} expected {c['expected']:<14} got {got:<14} {'OK' if got == c['expected'] else 'MISMATCH'}")
    print("\nall canaries match" if not bad else f"\n{bad} canary verdicts differ from the plan")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

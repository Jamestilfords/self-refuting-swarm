"""E3 scoring: run every proposal on validation and measure validation calibration.

    python -m modal_jobs.score_stream plan
    python -m modal_jobs.score_stream run --streams canary,baseline     # smoke: a handful of proposals
    python -m modal_jobs.score_stream run                               # everything; reruns skip finished work

Three functions with different access:
  score_val_batch   runs proposal code; mounts only the open volume.
  calibrate_cell    runs no proposal code; reads validation labels from the sealed volume (read-only) for gate G5.
  score_test_batch  the sealer; runs proposal code with the sealed volume mounted. Off unless --seal is given.
                    Do not use it for the registered run until proposals execute in the isolated sandbox.
The ledger (one row per proposal and cell, slide 9) is written locally to runs/ledger.jsonl.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import modal

from modal_jobs.app import app, open_volume, sealed_volume, settings
from modal_jobs.config import OPEN_MOUNT, SEALED_MOUNT

REPO = Path(__file__).resolve().parents[1]
ER1_SOURCE = REPO / "third_party" / "er1" / "src"


def _snapshot(src: Path) -> Path:
    """Freeze proposals/ into a temp folder, so a Proposer that is still writing cannot change files mid-upload.

    The plan, the container image and the ledger all use this one frozen copy.
    """
    dst = Path(tempfile.mkdtemp(prefix="skeptic_proposals_"))
    for f in sorted(src.iterdir()):
        if f.suffix in (".py", ".json"):
            shutil.copy2(f, dst / f.name)
    return dst


PROPOSALS = _snapshot(REPO / "proposals") if (REPO / "proposals").is_dir() else REPO / "proposals"
SCORE_OPTIONS = {"cpu": 8, "memory": 16 * 1024, "timeout": 60 * 60, "max_containers": 60}

score_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.5.4", "scipy==1.18.1", "scikit-learn==1.9.1")
    .env({**settings.container_env(), "SKEPTIC_ER1_SRC": "/opt/er1/src"})
)
if ER1_SOURCE.is_dir():       # baked in when launching locally; remote imports already have these paths
    score_image = score_image.add_local_dir(ER1_SOURCE, "/opt/er1/src", copy=True)
if PROPOSALS.is_dir():
    score_image = score_image.add_local_dir(PROPOSALS, "/opt/proposals", copy=True)
score_image = score_image.add_local_python_source("skeptic", "modal_jobs")


@app.function(image=score_image, volumes={OPEN_MOUNT: open_volume}, **SCORE_OPTIONS)
def score_val_batch(task: dict) -> dict:
    """Validation only. The sealed volume is not mounted here."""
    from skeptic import runner
    open_volume.reload()
    rows = runner.score_validation("/opt/proposals", task["shas"], OPEN_MOUNT, task["group"])
    open_volume.commit()
    return {"cell": task["cell"], "rows": rows}


@app.function(image=score_image, cpu=2, memory=4 * 1024, timeout=30 * 60,
              volumes={OPEN_MOUNT: open_volume.with_mount_options(read_only=True),
                       SEALED_MOUNT: sealed_volume.with_mount_options(read_only=True)})
def calibrate_cell(task: dict) -> dict:
    """Validation calibration for G5. No proposal code runs here, and both volumes are read-only."""
    from skeptic import runner
    rows = runner.validation_calibration(OPEN_MOUNT, SEALED_MOUNT, task["group"], task["items"])
    return {"cell": task["cell"], "rows": rows}


@app.function(image=score_image, volumes={OPEN_MOUNT: open_volume, SEALED_MOUNT: sealed_volume}, **SCORE_OPTIONS)
def score_test_batch(task: dict) -> dict:
    """The sealer: test scores go into the sealed volume; only statuses come back."""
    from skeptic import runner
    open_volume.reload()
    sealed_volume.reload()
    rows = runner.score_test("/opt/proposals", task["shas"], OPEN_MOUNT, SEALED_MOUNT, task["group"])
    sealed_volume.commit()
    return {"cell": task["cell"], "rows": rows}


def _collect(fn, tasks, label: str) -> dict:
    out, done, failed = {}, 0, 0
    for result in fn.map(tasks, order_outputs=False, return_exceptions=True):
        if isinstance(result, BaseException):
            failed += 1
            print(f"[{label}] a batch failed: {type(result).__name__}: {str(result)[:200]}", flush=True)
            continue
        done += 1
        for row in result["rows"]:
            out[(row["sha"], result["cell"])] = row
        if done % 10 == 0 or done + failed == len(tasks):
            print(f"[{label}] {done + failed}/{len(tasks)} batches", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    from skeptic import runner

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--streams", default=None, help="comma-separated: canary,baseline,templated,llm")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N proposals")
    parser.add_argument("--seal", action="store_true", help="also run the sealer (proposal code with /sealed mounted)")
    parser.add_argument("--skip-test", action="store_true", help="accepted for old commands; the sealer is off by default")
    args = parser.parse_args(argv)

    found = runner.discover(PROPOSALS, set(args.streams.split(",")) if args.streams else None)
    tasks = runner.build_tasks(found, args.limit)
    n = len({s for t in tasks for s in t["shas"]})
    print(f"{n} proposals x {len(runner.CELLS)} cells = {n * len(runner.CELLS)} ledger rows, in {len(tasks)} batches")
    if args.command == "plan":
        return 0

    with modal.enable_output():
        with app.run(environment_name=settings.environment):
            val_rows = _collect(score_val_batch, tasks, "validation")
            ok = lambda t: [s for s in t["shas"] if val_rows.get((s, t["cell"]), {}).get("status") == "ok"]
            cal_tasks = [{"cell": cell, "group": group,
                          "items": [(s, t["arm"]) for t in tasks if t["cell"] == cell for s in ok(t)]}
                         for cell, group in runner.CELLS.items()]
            cal_rows = _collect(calibrate_cell, [t for t in cal_tasks if t["items"]], "calibration")
            test_rows = {}
            if args.seal:
                test_tasks = [{**t, "shas": ok(t)} for t in tasks if ok(t)]
                test_rows = _collect(score_test_batch, test_tasks, "sealer")

    ledger = runner.merge_ledger(found, tasks, val_rows, cal_rows, test_rows)
    partial = args.streams or args.limit
    path = REPO / "runs" / ("ledger_partial.jsonl" if partial else "ledger.jsonl")
    runner.write_ledger(ledger, path)
    print(runner.summarize(ledger))
    with_probs = [r for r in ledger if r["status"] == "ok" and r["has_probs"]]
    print(f"calibration: {sum(r['val']['ece'] is not None for r in with_probs)} of {len(with_probs)} rows that return probs have a validation ECE")
    print(f"wrote {len(ledger)} rows to {path.relative_to(REPO)}")
    for row in ledger:
        if row["stream"] == "canary" and row["cell"] == "clinc150/bert":
            auroc = row["val"]["auroc"] if row["val"] else None
            print("canary", json.dumps({"sha": row["sha"], "status": row["status"],
                                        "val_auroc_seed0": auroc[0] if auroc else None}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

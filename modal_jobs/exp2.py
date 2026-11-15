"""Experiment 2 driver: closed-loop runs with validation scoring on Modal.

    python -m modal_jobs.exp2 plan
    python -m modal_jobs.exp2 run --smoke          # 3 arms x 3 proposals on Banking77-OOS/BERT, into runs/exp2_smoke
    python -m modal_jobs.exp2 run                  # registered design: 2 cells x 3 arms x 3 runs x 80 proposals
    python -m modal_jobs.exp2 report

Each run is: Proposer (LLM) -> score_text on Modal -> ratchet or critic (local, code only) -> memory -> next.
Rerunning the same command resumes every run where it stopped.

Access:
  score_text      runs proposal code; mounts only the open volume, read-only.
  validation_ece  runs no proposal code; reads validation labels from the sealed volume, read-only (gate G5).
The registered run refuses to start while PREREGISTRATION.md is still marked DRAFT, because the Proposer
reacts to verdicts.
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

from modal_jobs.app import open_volume, sealed_volume, settings
from modal_jobs.config import OPEN_MOUNT, SEALED_MOUNT

REPO = Path(__file__).resolve().parents[1]
ER1_SOURCE = REPO / "third_party" / "er1" / "src"
CELLS = {"banking77_oos/bert": "banking77_bert", "clinc150/bert": "clinc150_bert"}     # BERT only (registered)
exp2_app = modal.App(f"{settings.app_name}-exp2")

score_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.5.4", "scipy==1.18.1", "scikit-learn==1.9.1")
    .env({**settings.container_env(), "SKEPTIC_ER1_SRC": "/opt/er1/src"})
)
if ER1_SOURCE.is_dir():       # baked in when launching locally; remote imports already have these paths
    score_image = score_image.add_local_dir(ER1_SOURCE, "/opt/er1/src", copy=True)
score_image = score_image.add_local_python_source("skeptic", "modal_jobs")

_CTX: dict = {}               # caches loaded per container, reused across calls


@exp2_app.function(image=score_image, cpu=4, memory=8 * 1024, timeout=15 * 60, max_containers=12,
                   volumes={OPEN_MOUNT: open_volume.with_mount_options(read_only=True)})
def score_text(group: str, text: str) -> dict:
    """Validation scores of one proposal on the five seeds of one cell. The sealed volume is not mounted."""
    from skeptic import loop_scoring
    return loop_scoring.score_text(text, OPEN_MOUNT, group, _CTX)


@exp2_app.function(image=score_image, cpu=1, memory=2 * 1024, timeout=5 * 60,
                   volumes={SEALED_MOUNT: sealed_volume.with_mount_options(read_only=True)})
def validation_ece(group: str, arm: str, pred, conf, n_in: int) -> list:
    """Validation ECE for G5. No proposal code runs here."""
    from skeptic import loop_scoring
    return loop_scoring.validation_ece(SEALED_MOUNT, group, arm, pred, conf, n_in)


def _retrying(call, tries: int = 3):
    from skeptic.agent_runtime import BudgetExhausted

    def wrapped(*args):
        for attempt in range(tries):
            try:
                return call(*args)
            except BudgetExhausted:
                raise
            except Exception:
                if attempt == tries - 1:
                    raise
                time.sleep(5 * (attempt + 1))
    return wrapped


def _scorer(group: str):
    from skeptic import api
    remote_score, remote_ece = _retrying(score_text.remote), _retrying(validation_ece.remote)

    def score(text: str) -> dict:
        out = remote_score(group, text)
        if out["status"] != "ok":
            raise api.Malformed(out["status"], out.get("detail", ""))
        ece = remote_ece(group, out["arm"], out["pred"], out["conf"], out["n_in"]) if "conf" in out else None
        return {"scores": out["scores"], "n_in": out["n_in"], "ece": ece}
    return score


def _run_one(spec, args, cells) -> tuple[str, str]:
    from skeptic.agent_runtime import BudgetExhausted, RunBudget
    from skeptic.analysis import exp2
    from skeptic.memory import get_memory
    from skeptic.streams.proposer import openai_llm
    from skeptic.streams.scout import Scout

    cell, arm, k = spec
    group = CELLS[cell]
    tag = f"{group}/{arm}{k}"
    log = lambda message: print(f"[{tag}] {message}", flush=True)
    try:
        run_dir = args.out / group / f"{arm}{k}"
        (run_dir / "proposals").mkdir(parents=True, exist_ok=True)
        if not (run_dir / "proposals" / "_template.py").exists():
            shutil.copy(REPO / "proposals" / "_template.py", run_dir / "proposals" / "_template.py")
        budget = RunBudget(run_dir / "budget.json", max_calls=1000, max_searches=40,
                           max_tokens=8_000_000, max_output_tokens=2048)
        budget.save()
        llm = budget.llm(_retrying(openai_llm(args.model)))
        memory = get_memory(run_dir / "memory.jsonl", backend=args.memory_backend, namespace=f"exp2-{group}-{arm}{k}")
        scout = None
        if arm == "B" and not args.no_scout and os.environ.get("EXA_API_KEY"):
            from exa_py import Exa
            scout = Scout(memory, llm, Exa(api_key=os.environ["EXA_API_KEY"], timeout=30), num_results=3,
                          before_search=budget.search, state_path=run_dir / "scout.json")
        loop = exp2.Loop(arm, cells[cell], run_dir, llm=llm, score=_scorer(group), model=args.model, memory=memory,
                         scout=scout, analyst_llm=llm, analyst_every=args.analyst_every, log=log)
        loop.run(args.n)
        return tag, f"done: {len(loop.rows)} slots, tokens {budget.used['input_tokens']}+{budget.used['output_tokens']}"
    except BudgetExhausted as e:
        return tag, f"stopped: {e}"
    except Exception as e:       # one failed run must not stop the others; rerun the command to resume it
        log(traceback.format_exc(limit=4))
        return tag, f"failed: {type(e).__name__}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("plan", "run", "report"))
    parser.add_argument("--smoke", action="store_true", help="3 proposals per arm on Banking77-OOS/BERT, into runs/exp2_smoke")
    parser.add_argument("--cells", default=",".join(CELLS))
    parser.add_argument("--arms", default="A,B,C")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--n", type=int, default=80, help="proposal slots per run")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--workers", type=int, default=18, help="runs in parallel")
    parser.add_argument("--analyst-every", type=int, default=25)
    parser.add_argument("--memory-backend", default="auto", choices=("auto", "local", "pastdev"))
    parser.add_argument("--no-scout", action="store_true")
    parser.add_argument("--scores", type=Path, default=REPO / "runs" / "scores")
    parser.add_argument("--ledger", type=Path, default=REPO / "runs" / "ledger.jsonl")
    args = parser.parse_args(argv)
    if args.smoke:
        args.cells, args.runs, args.n, args.analyst_every = "banking77_oos/bert", 1, 3, 2
    args.out = args.out or REPO / "runs" / ("exp2_smoke" if args.smoke else "exp2")
    cells, arms = args.cells.split(","), args.arms.split(",")
    if set(cells) - set(CELLS) or set(arms) - {"A", "B", "C"} or args.runs < 1 or args.n < 1:
        parser.error(f"cells must be from {sorted(CELLS)} and arms from A,B,C")
    specs = [(cell, arm, k) for cell in cells for arm in arms for k in range(args.runs)]

    from skeptic.analysis import exp2
    if args.command == "report":
        exp2.report(args.out)
        return 0
    print(f"{len(specs)} runs x {args.n} proposal slots = {len(specs) * args.n} proposals, into {args.out}")
    if args.command == "plan":
        return 0

    status = (REPO / "PREREGISTRATION.md").read_text(encoding="utf-8")[:600]
    if "DRAFT" in status and not args.smoke:
        parser.error("PREREGISTRATION.md is still marked DRAFT. Experiment 2 starts only after it is registered; use --smoke to test.")

    from skeptic.agents import load_environment
    from skeptic.critic_inputs import load_cell
    from skeptic.rules.critic_bridge import ensure_er1_path
    load_environment(REPO / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY is not set in .env")
    ensure_er1_path()
    exp2.use_fast_bootstrap()
    loaded = {cell: load_cell(cell, args.scores, args.ledger, REPO / "proposals" / "baselines.json") for cell in cells}

    with modal.enable_output():
        with exp2_app.run(environment_name=settings.environment):
            with ThreadPoolExecutor(max_workers=max(1, min(args.workers, len(specs)))) as pool:
                results = list(pool.map(lambda spec: _run_one(spec, args, loaded), specs))
    for tag, outcome in results:
        print(f"{tag}: {outcome}")
    exp2.report(args.out)
    return int(any(not outcome.startswith("done") for _, outcome in results))


if __name__ == "__main__":
    raise SystemExit(main())

"""Bounded Scout -> memory -> Proposer workflow. Use plan/check before run."""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from skeptic.agent_runtime import BudgetExhausted, ConfigurationError, RunBudget, atomic_json

ENV_KEYS = (
    "EXA_API_KEY", "PASTDEV_API_KEY", "PAST_API_KEY", "OPENAI_API_KEY",
    "SKEPTIC_PROPOSER_MODEL", "SKEPTIC_MEMORY_BACKEND", "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET", "MODAL_PROFILE", "MODAL_ENVIRONMENT", "SKEPTIC_APP_NAME",
)


def load_environment(path: Path) -> None:
    """Read only the selected file, without interpolation or overriding shell settings."""
    from dotenv import dotenv_values
    for key, value in dotenv_values(path, interpolate=False).items():
        if key in ENV_KEYS and value:
            os.environ.setdefault(key, value)


def readiness(args) -> dict:
    """Local presence checks only. This does not validate credentials with providers."""
    checks = {"OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
              "SKEPTIC_PROPOSER_MODEL": bool(args.model)}
    if not args.no_scout:
        checks["EXA_API_KEY"] = bool(os.environ.get("EXA_API_KEY"))
    if args.memory_backend == "pastdev":
        checks["PASTDEV_API_KEY"] = bool(os.environ.get("PASTDEV_API_KEY") or os.environ.get("PAST_API_KEY"))
    token_id, token_secret = os.environ.get("MODAL_TOKEN_ID"), os.environ.get("MODAL_TOKEN_SECRET")
    if token_id or token_secret:
        # Never mix one supplied token half with an unrelated profile's other half.
        checks["MODAL_TOKEN_ID"] = bool(token_id)
        checks["MODAL_TOKEN_SECRET"] = bool(token_secret)
    else:
        from modal.config import config
        checks["Modal profile credentials"] = bool(config.get("token_id") and config.get("token_secret"))
    for module in ("openai", "exa_py"):
        checks[f"dependency:{module}"] = importlib.util.find_spec(module) is not None
    checks["pinned ER-1 source"] = (Path(__file__).resolve().parents[1] / "third_party/er1/src/bert_daedl").is_dir()
    return {"ready": all(checks.values()), "checks": checks,
            "missing": [key for key, ok in checks.items() if not ok],
            "credential_validation": "local presence only; no service contacted"}


@contextmanager
def run_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ConfigurationError("Another agent process is using this run directory") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def run(args) -> int:
    from exa_py import Exa
    from modal_jobs.agent_validation import modal_validator
    from skeptic.memory import get_memory
    from skeptic.streams.proposer import Proposer, openai_llm, smoke_ctx
    from skeptic.streams.scout import Scout

    root = args.run_dir
    config = {key: getattr(args, key) for key in (
        "model", "memory_backend", "no_scout", "scout_every", "num_results",
        "max_attempts", "max_calls", "max_searches", "budget_tokens", "max_output_tokens")}
    config["validation"] = "synthetic-modal-v1"
    with run_lock(root):
        session_path = root / "session.json"
        if session_path.exists():
            session = json.loads(session_path.read_text())
            if session["config"] != config:
                raise ConfigurationError("Resume configuration differs; use the original settings or a new run directory")
            if not (root / "budget.json").is_file():
                raise ConfigurationError("Saved budget is missing; restore the complete run directory before resuming")
        else:
            if any(root.glob("*.json*")) or (root / "proposals").exists():
                raise ConfigurationError("Run directory contains outputs without a session; use a new directory")
            session = {"run_id": f"skeptic-agent-{uuid4().hex}", "config": config}
            atomic_json(session_path, session)
        budget = RunBudget(root / "budget.json", max_calls=args.max_calls, max_searches=args.max_searches,
                           max_tokens=args.budget_tokens, max_output_tokens=args.max_output_tokens)
        budget.save()
        manifest = root / "proposals/llm.json"
        old = json.loads(manifest.read_text()) if manifest.exists() else {}
        completed = len(old.get("proposals", [])) + len(old.get("rejected", []))
        if completed >= args.n:
            print(f"Target already reached: {completed} proposals attempted; no calls made")
            return 0
        memory = get_memory(root / "memory.jsonl", backend=args.memory_backend, namespace=session["run_id"])
        llm = budget.llm(openai_llm(args.model, max_output_tokens=args.max_output_tokens))
        scout = None if args.no_scout else Scout(
            memory, llm, Exa(api_key=os.environ["EXA_API_KEY"], timeout=30),
            num_results=args.num_results, before_search=budget.search, state_path=root / "scout.json")
        ctx = smoke_ctx()
        proposer = None
        stop = "target_reached"
        with modal_validator(ctx) as validator:
            proposer = Proposer(memory, llm, ctx, root / "proposals", args.model,
                                "synthetic-modal", args.max_attempts, validator=validator)
            try:
                proposer.run(args.n - completed, scout=scout, scout_every=args.scout_every,
                             budget_tokens=args.budget_tokens)
            except BudgetExhausted as exc:
                stop = "budget_exhausted"
                print(f"Stopped: {exc}")
            finally:
                proposer.save()
        report = {"stop": stop, "accepted": len(proposer.rows), "rejected": len(proposer.rejected),
                  "usage": budget.used, "memory_writers": memory.stats(),
                  "pastdev_ingested_this_process": getattr(memory, "ingested", 0),
                  "pastdev_errors_this_process": getattr(memory, "errors", 0),
                  "validation": "synthetic checks only; these are not critic keep/reject decisions"}
        atomic_json(root / "report.json", report)
        print(json.dumps(report, indent=2))
        return 3 if stop == "budget_exhausted" else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "check", "run"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/agents/development"))
    parser.add_argument("--model")
    parser.add_argument("--memory-backend", choices=("local", "pastdev"))
    parser.add_argument("--n", type=int, default=5, help="Total proposal attempts, including previously completed ones")
    parser.add_argument("--no-scout", action="store_true")
    parser.add_argument("--scout-every", type=int, default=10)
    parser.add_argument("--num-results", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--max-calls", type=int, default=20)
    parser.add_argument("--max-searches", type=int, default=2)
    parser.add_argument("--budget-tokens", type=int, default=150_000)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    args = parser.parse_args(argv)
    if min(args.n, args.scout_every, args.max_attempts, args.max_calls, args.budget_tokens, args.max_output_tokens) < 1 \
            or args.max_searches < 0 or not 1 <= args.num_results <= 10:
        parser.error("Counts and limits must be positive; max-searches may be zero and num-results must be 1..10")
    try:
        load_environment(args.env_file)
        args.model = args.model or os.environ.get("SKEPTIC_PROPOSER_MODEL")
        args.memory_backend = args.memory_backend or os.environ.get("SKEPTIC_MEMORY_BACKEND", "local")
        if args.memory_backend not in ("local", "pastdev"):
            raise ConfigurationError("SKEPTIC_MEMORY_BACKEND must be local or pastdev")
        status = readiness(args)
        if args.command in ("plan", "check"):
            print(json.dumps({"command": args.command, "readiness": status,
                              "run_dir": str(args.run_dir), "model": args.model,
                              "memory_backend": args.memory_backend, "total_proposal_target": args.n,
                              "limits": {"llm_calls": args.max_calls, "exa_searches": args.max_searches,
                                         "tokens": args.budget_tokens, "output_tokens_per_call": args.max_output_tokens},
                              "workflow": ([] if args.no_scout else ["Exa research", "source memory"])
                                          + ["LLM proposal", "Modal synthetic checks"],
                              "compute": "local coordinator; Modal CPU sandboxes; no training or dataset scoring"}, indent=2))
            return 0 if args.command == "plan" or status["ready"] else 2
        if not status["ready"]:
            print("Missing configuration: " + ", ".join(status["missing"]))
            return 2
        return run(args)
    except KeyboardInterrupt:
        print("Interrupted; saved usage remains charged. Resume with the same run directory.")
        return 130
    except ConfigurationError as exc:
        print(str(exc))
        return 2
    except Exception as exc:
        # SDK exception strings and HTTP response bodies can contain credentials.
        print(f"Agent workflow stopped ({type(exc).__name__}). Check configuration and saved run files.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

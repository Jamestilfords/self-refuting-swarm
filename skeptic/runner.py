"""Runner and sealer core (E3/E4 scoring), independent of Modal so it can be tested locally.

score_validation  runs each proposal on validation for the five seeds of one cell. It reads only the
                  open store and writes per-item validation scores there.
score_test        the sealer. Runs the same files on test, writes scores into the sealed store, and
                  returns a status plus validation calibration. It never returns a test-derived number.

Layout written:
    <open>/scores/<group>/<sha>.npz     scores (5, n_val + n_oos_val) float32, n_in; pred, conf if probs
    <sealed>/scores/<group>/<sha>.npz   the same for test + oos_test
In every array the first n_in items are in-scope and the rest are out-of-scope.
"""
from __future__ import annotations

import signal
import time
from pathlib import Path

import numpy as np
from scipy.special import softmax

from skeptic import api

CELLS = {"clinc150/bert": "clinc150_bert", "clinc150/roberta": "clinc150_roberta",
         "banking77_oos/bert": "banking77_bert", "banking77_oos/roberta": "banking77_roberta"}
SEEDS = tuple(range(5))
HARD_LIMIT_S = 150          # kills a hung proposal; the 60 s rule itself is checked by api.run_proposal
FIELDS = ("logits", "feats", "pooled")


class _Limit:
    """Raise Malformed('timeout') if the block runs past HARD_LIMIT_S. A no-op off the main thread."""

    def __enter__(self):
        def stop(signum, frame):
            raise api.Malformed("timeout", f"killed after {HARD_LIMIT_S}s")
        try:
            self.old = signal.signal(signal.SIGALRM, stop)
            signal.alarm(HARD_LIMIT_S)
            self.armed = True
        except (ValueError, AttributeError):
            self.armed = False
        return self

    def __exit__(self, *exc):
        if self.armed:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self.old)
        return False


# ------------------------------------------------------------------ metrics (validation only)
def auroc(inside, outside) -> float:
    """P(OOS score > in-scope score), ties counted as half."""
    inside, outside = np.asarray(inside, np.float64), np.asarray(outside, np.float64)
    ordered = np.sort(np.concatenate([inside, outside]))
    left, right = np.searchsorted(ordered, outside, "left"), np.searchsorted(ordered, outside, "right")
    rank_sum = ((left + right + 1) / 2).sum()
    return float((rank_sum - len(outside) * (len(outside) + 1) / 2) / (len(inside) * len(outside)))


def fpr95(inside, outside) -> float:
    """In-scope false-alarm rate at 95% OOS recall (ER-1 stats.fpr_at_95)."""
    return float(np.mean(np.asarray(inside) >= np.quantile(np.asarray(outside), 0.05)))


def ece(conf, correct, n_bins: int = 15) -> float:
    """Expected calibration error (ER-1 eval.metrics.ece)."""
    conf, correct = np.asarray(conf, np.float64), np.asarray(correct, bool)
    edges, total = np.linspace(0.0, 1.0, n_bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


# ------------------------------------------------------------------ cache loading
def open_ctx(run_dir):
    """(ctx, n_val) from an open-store run directory. Eval is val followed by oos_val."""
    c = np.load(Path(run_dir) / "cache.npz")
    part = lambda s: {k: c[f"{s}__{k}"] for k in FIELDS}
    val, oos = part("val"), part("oos_val")
    ctx = api.make_ctx({**part("train"), "labels": c["train__labels"]}, {k: np.concatenate([val[k], oos[k]]) for k in FIELDS})
    return ctx, len(val["logits"])


def sealed_run(run_dir) -> dict:
    """Test context, validation labels and validation logits from a sealed run directory."""
    c, p = np.load(Path(run_dir) / "cache.npz"), np.load(Path(run_dir) / "pooled.npz")
    part = lambda s: {"logits": c[f"{s}__logits"], "feats": c[f"{s}__feats"], "pooled": p[f"{s}__pooled"]}
    test, oos = part("test"), part("oos_test")
    ctx = api.make_ctx({**part("train"), "labels": c["train__labels"]}, {k: np.concatenate([test[k], oos[k]]) for k in FIELDS})
    return {"ctx": ctx, "n_test": len(test["logits"]), "val_labels": c["val__labels"], "val_logits": c["val__logits"]}


def _seed_dirs(root, group: str, arm: str) -> list[Path]:
    return [Path(root) / group / f"{arm}_seed{s}" for s in SEEDS]


def _pack(per_seed: list[dict]) -> dict:
    out = {"scores": np.stack([r["scores"] for r in per_seed]).astype(np.float32)}
    if per_seed[0]["probs"] is not None:
        out["pred"] = np.stack([r["probs"].argmax(1) for r in per_seed]).astype(np.int16)
        out["conf"] = np.stack([r["probs"].max(1) for r in per_seed]).astype(np.float32)
    return out


def _val_metrics(scores: np.ndarray, n_in: int) -> dict:
    return {"auroc": [round(auroc(s[:n_in], s[n_in:]), 6) for s in scores],
            "fpr95": [round(fpr95(s[:n_in], s[n_in:]), 6) for s in scores]}


def _run(p, ctx, full_checks: bool) -> dict:
    with _Limit():
        if full_checks:
            return api.validate(p, ctx)
        scores, probs, seconds = api.run_proposal(p, ctx)
        return {"scores": scores, "probs": probs, "runtime_s": round(seconds, 3)}


# ------------------------------------------------------------------ validation runner (open store only)
def score_validation(proposals_dir, shas, open_root, group: str) -> list[dict]:
    """One row per proposal for one cell. Never touches the sealed store."""
    out_dir = Path(open_root) / "scores" / group
    out_dir.mkdir(parents=True, exist_ok=True)
    caches: dict[str, list] = {}
    rows = []
    for sha in shas:
        row = {"sha": sha, "status": "ok", "family": None, "arm": None, "parents": [], "source": "",
               "memory_used": [], "runtime_s": None, "val": None, "has_probs": False}
        try:
            p = api.load_proposal(Path(proposals_dir) / f"{sha}.py")
            row.update(family=p.family, arm=p.arm, parents=list(p.parents), source=p.source,
                       memory_used=list(p.memory_used))
            target = out_dir / f"{sha}.npz"
            if target.exists():                                   # resume: reuse stored scores
                z = np.load(target)
                row.update(val=_val_metrics(z["scores"], int(z["n_in"])), has_probs="conf" in z.files,
                           runtime_s=float(z["runtime_s"]))
            else:
                if p.arm not in caches:
                    caches[p.arm] = [open_ctx(d) for d in _seed_dirs(open_root, group, p.arm)]
                t0 = time.perf_counter()
                per_seed = [_run(p, ctx, full_checks=(k == 0)) for k, (ctx, _) in enumerate(caches[p.arm])]
                n_in = caches[p.arm][0][1]
                packed = _pack(per_seed)
                runtime = round(max(r["runtime_s"] for r in per_seed), 3)
                np.savez_compressed(target, n_in=n_in, runtime_s=runtime, **packed)
                row.update(val=_val_metrics(packed["scores"], n_in), has_probs="conf" in packed, runtime_s=runtime,
                           wall_s=round(time.perf_counter() - t0, 1))
        except api.Malformed as e:
            row.update(status=e.reason, detail=e.detail[:200])
        except Exception as e:  # a broken proposal must not take the batch down
            row.update(status="runner_error", detail=f"{type(e).__name__}: {e}"[:200])
        rows.append(row)
    return rows


# ------------------------------------------------------------------ sealer (sealed store; status only)
def score_test(proposals_dir, shas, open_root, sealed_root, group: str) -> list[dict]:
    """Write test scores into the sealed store. Returns status and validation calibration only."""
    out_dir = Path(sealed_root) / "scores" / group
    out_dir.mkdir(parents=True, exist_ok=True)
    runs: dict[str, list] = {}
    rows = []
    for sha in shas:
        row = {"sha": sha, "sealed": "ok", "val_ece": None, "val_ece_plain": None}
        try:
            p = api.load_proposal(Path(proposals_dir) / f"{sha}.py")
            if p.arm not in runs:
                runs[p.arm] = [sealed_run(d) for d in _seed_dirs(sealed_root, group, p.arm)]
            target = out_dir / f"{sha}.npz"
            if not target.exists():
                per_seed = [_run(p, r["ctx"], full_checks=False) for r in runs[p.arm]]
                np.savez_compressed(target, n_in=runs[p.arm][0]["n_test"], **_pack(per_seed))
            # validation calibration: needs validation labels, which only the sealed store holds
            row["val_ece_plain"] = [round(ece(softmax(r["val_logits"].astype(np.float64), axis=1).max(1),
                                              r["val_logits"].argmax(1) == r["val_labels"]), 6) for r in runs[p.arm]]
            val = np.load(Path(open_root) / "scores" / group / f"{sha}.npz")
            if "conf" in val.files:
                n = int(val["n_in"])
                row["val_ece"] = [round(ece(val["conf"][k, :n], val["pred"][k, :n] == r["val_labels"]), 6)
                                  for k, r in enumerate(runs[p.arm])]
        except api.Malformed as e:
            row.update(sealed=e.reason)
        except Exception as e:
            row.update(sealed="sealer_error", detail=f"{type(e).__name__}: {e}"[:200])
        rows.append(row)
    return rows


# ------------------------------------------------------------------ planning and the ledger (run locally)
BATCH = 6                                      # proposals per container call; one arm per batch
_STREAM_ORDER = ("canaries", "templated", "llm", "baselines")
_STREAM_TAG = {"canaries": "canary", "templated": "templated", "llm": "llm", "baselines": "baseline"}


def discover(proposals_dir, streams=None) -> dict:
    """{sha: {"stream", "arm"}} from the manifests in proposals/. A file in two manifests keeps the first tag."""
    import json
    out: dict = {}
    for name in _STREAM_ORDER:
        path = Path(proposals_dir) / f"{name}.json"
        if not path.exists() or (streams and _STREAM_TAG[name] not in streams):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in (data["proposals"] if isinstance(data, dict) else data):
            sha = row["sha"]
            if sha in out or not (Path(proposals_dir) / f"{sha}.py").exists():
                continue
            try:
                arm = api.static_check((Path(proposals_dir) / f"{sha}.py").read_text(encoding="utf-8"))["ARM"]
            except api.Malformed:
                arm = "ce"                     # e.g. the leak canary: the runner records why it is malformed
            out[sha] = {"stream": _STREAM_TAG[name], "arm": arm}
    return out


def build_tasks(found: dict, limit: int | None = None) -> list[dict]:
    shas = sorted(found)[:limit] if limit else sorted(found)
    tasks = []
    for cell, group in CELLS.items():
        for arm in ("ce", "edl"):
            mine = [s for s in shas if found[s]["arm"] == arm]
            tasks += [{"cell": cell, "group": group, "arm": arm, "shas": mine[i:i + BATCH]}
                      for i in range(0, len(mine), BATCH)]
    return tasks


def merge_ledger(found: dict, tasks: list[dict], val_rows: dict, test_rows: dict) -> list[dict]:
    """One ledger row per proposal and cell (slide 9). val_rows/test_rows are keyed by (sha, cell)."""
    ledger = []
    for t in tasks:
        for sha in t["shas"]:
            key = (sha, t["cell"])
            v = val_rows.get(key) or {"sha": sha, "status": "infra_error", "val": None, "arm": t["arm"]}
            s = test_rows.get(key, {})
            val = dict(v["val"]) if v.get("val") else None
            if val is not None:
                val["ece"], val["ece_plain"] = s.get("val_ece"), s.get("val_ece_plain")
            ledger.append({"sha": sha, "stream": found[sha]["stream"], "family": v.get("family"), "arm": v.get("arm"),
                           "cell": t["cell"], "status": v["status"], "detail": v.get("detail"), "val": val,
                           "parents": v.get("parents", []), "source": v.get("source", ""),
                           "memory_used": v.get("memory_used", []), "runtime_s": v.get("runtime_s"),
                           "sealed": s.get("sealed", "not_run" if v["status"] == "ok" else "skipped")})
    return sorted(ledger, key=lambda r: (r["sha"], r["cell"]))


def write_ledger(rows: list[dict], path="runs/ledger.jsonl") -> None:
    import json
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def summarize(rows: list[dict]) -> str:
    from collections import Counter
    lines = []
    for stream in sorted({r["stream"] for r in rows}):
        counts = Counter(r["status"] for r in rows if r["stream"] == stream)
        lines.append(f"{stream:<10} " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    sealed = Counter(r["sealed"] for r in rows)
    lines.append("sealed     " + ", ".join(f"{k} {v}" for k, v in sorted(sealed.items())))
    return "\n".join(lines)

"""Container-side validation scoring of one proposal text, for the closed loop (Experiment 2).

score_text      runs the proposal on the five seeds of one cell and returns per-item validation scores.
                It needs only the open store and writes nothing.
validation_ece  validation ECE for gate G5 from returned confidences. It runs no proposal code and reads
                only validation labels from the sealed store.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from skeptic import api, runner


def score_text(text: str, open_root, group: str, cache: dict | None = None) -> dict:
    """{"status": "ok", "arm", "n_in", "scores" (5, n) float32, "pred"/"conf" if probs} or {"status": reason, "detail"}."""
    cache = {} if cache is None else cache
    try:
        if len(text.encode("utf-8")) > 256_000:
            raise api.Malformed("source", "proposal exceeds the source size limit")
        arm = api.static_check(text)["ARM"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"{api.proposal_sha(text)}.py"
            path.write_text(text, encoding="utf-8")
            proposal = api.load_proposal(path)
        key = (group, arm)
        if key not in cache:
            cache[key] = [runner.open_ctx(d) for d in runner._seed_dirs(open_root, group, arm)]
        per_seed = [runner._run(proposal, ctx, full_checks=(k == 0)) for k, (ctx, _) in enumerate(cache[key])]
        return {"status": "ok", "arm": arm, "n_in": int(cache[key][0][1]),
                "runtime_s": round(max(r["runtime_s"] for r in per_seed), 3), **runner._pack(per_seed)}
    except api.Malformed as e:
        return {"status": e.reason, "detail": e.detail[:300]}
    except Exception as e:  # a broken proposal must not take the loop down
        return {"status": "runtime_error", "detail": f"{type(e).__name__}: {e}"[:300]}


def validation_ece(sealed_root, group: str, arm: str, pred, conf, n_in: int) -> list[float]:
    """Per-seed validation ECE (15 bins). No proposal code runs here; only validation labels are read."""
    from skeptic.store import SealedStore
    if arm not in ("ce", "edl"):
        raise ValueError("Unknown arm")
    store = SealedStore(sealed_root, Path.cwd())
    pred, conf = np.asarray(pred), np.asarray(conf, np.float64)
    out = []
    for k in range(len(pred)):
        labels = store.read_array(f"{group}/{arm}_seed{k}", "val", "labels")
        if labels.shape != (n_in,):
            raise ValueError("Validation labels and scores are misaligned")
        out.append(round(runner.ece(conf[k, :n_in], pred[k, :n_in] == labels), 6))
    return out

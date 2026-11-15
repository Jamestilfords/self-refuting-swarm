"""E3 worker and wire format for a proposal in a disposable Modal Sandbox.

Only the trusted Modal wrapper prepares input and interprets output. This
module is also importable locally for protocol tests; importing it never runs
proposal code. The worker process must be launched inside the Sandbox.
"""

from __future__ import annotations

import io
import json
import math
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

from skeptic import api

INPUT_KEYS = frozenset({
    "train__logits", "train__feats", "train__pooled", "train__labels",
    "eval__logits", "eval__feats", "eval__pooled",
})
OUTPUT_KEYS = frozenset({"scores", "probs", "runtime_s"})


def pack_input(train: dict, eval: dict) -> bytes:
    """Serialize just one cell's train and anonymous evaluation arrays."""
    values = {f"train__{key}": train[key] for key in ("logits", "feats", "pooled", "labels")}
    values.update({f"eval__{key}": eval[key] for key in ("logits", "feats", "pooled")})
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **values)
    return buffer.getvalue()


def unpack_input(data: bytes) -> api.Ctx:
    with np.load(io.BytesIO(data), allow_pickle=False) as payload:
        if set(payload.files) != INPUT_KEYS:
            raise ValueError("Sandbox input has unexpected arrays")
        train = {key: payload[f"train__{key}"] for key in ("logits", "feats", "pooled", "labels")}
        eval_data = {key: payload[f"eval__{key}"] for key in ("logits", "feats", "pooled")}
    return api.make_ctx(train, eval_data)


def unpack_output(data: bytes, n: int, k: int, *, require_variation: bool = True) -> tuple[np.ndarray, float]:
    """Validate untrusted worker output again in the trusted wrapper."""
    limit = 8 * n * (k + 1) + 4096
    if len(data) > limit:
        raise api.Malformed("sandbox_error", "Oversized worker output")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            infos = archive.infolist()
            if len(infos) != len(OUTPUT_KEYS) \
                    or {info.filename for info in infos} != {f"{key}.npy" for key in OUTPUT_KEYS} \
                    or sum(info.file_size for info in infos) > limit:
                raise api.Malformed("sandbox_error", "Unexpected worker archive")
    except zipfile.BadZipFile as exc:
        raise api.Malformed("sandbox_error", "Invalid worker archive") from exc
    with np.load(io.BytesIO(data), allow_pickle=False) as payload:
        if set(payload.files) != OUTPUT_KEYS:
            raise api.Malformed("runtime_error", "Unexpected worker output")
        scores = np.asarray(payload["scores"], np.float64)
        probs = np.asarray(payload["probs"], np.float64)
        runtime = np.asarray(payload["runtime_s"], np.float64)
    if scores.shape != (n,):
        raise api.Malformed("length")
    if not np.isfinite(scores).all():
        raise api.Malformed("nan")
    if require_variation and np.ptp(scores) == 0:
        raise api.Malformed("constant")
    if probs.shape != (0,) and (probs.shape != (n, k) or not np.isfinite(probs).all()
                               or (probs < 0).any() or not np.allclose(probs.sum(1), 1.0, atol=1e-4)):
        raise api.Malformed("probs")
    if runtime.shape != () or not math.isfinite(float(runtime)) or not 0 <= float(runtime) <= api.MAX_SECONDS:
        raise api.Malformed("runtime_error")
    return scores, float(runtime)


def _score_subset(proposal: api.Proposal, ctx: api.Ctx) -> tuple[np.ndarray, np.ndarray | None, float]:
    """The subset pass allows constant subsets, as api.validate does."""
    start = time.perf_counter()
    try:
        out = proposal.score(ctx)
    except Exception as exc:
        raise api.Malformed("runtime_error", type(exc).__name__) from exc
    seconds = time.perf_counter() - start
    if seconds > api.MAX_SECONDS:
        raise api.Malformed("timeout")
    probs = None
    if isinstance(out, tuple):
        if len(out) != 2:
            raise api.Malformed("length")
        out, probs = out
    try:
        scores = np.asarray(out, np.float64)
    except (TypeError, ValueError) as exc:
        raise api.Malformed("length") from exc
    if scores.shape != (len(ctx.eval),):
        raise api.Malformed("length")
    if not np.isfinite(scores).all():
        raise api.Malformed("nan")
    if probs is not None:
        probs = np.asarray(probs, np.float64)
        k = ctx.train.logits.shape[1]
        if probs.shape != (len(ctx.eval), k) or not np.isfinite(probs).all() \
                or (probs < 0).any() or not np.allclose(probs.sum(1), 1.0, atol=1e-4):
            raise api.Malformed("probs")
    return scores, probs, seconds


def worker(proposal_path: str, input_path: str, output_path: str, status_path: str,
           *, subset: bool = False) -> None:
    """Execute untrusted proposal code. Call only from a disposable Sandbox."""
    status = "ok"
    try:
        proposal = api.load_proposal(proposal_path)
        ctx = unpack_input(Path(input_path).read_bytes())
        scores, probs, seconds = (_score_subset(proposal, ctx) if subset
                                  else api.run_proposal(proposal, ctx))
        np.savez_compressed(output_path, scores=scores,
                            probs=probs if probs is not None else np.empty(0),
                            runtime_s=np.asarray(seconds))
    except api.Malformed as exc:
        status = exc.reason
    except Exception:
        status = "runtime_error"
    Path(status_path).write_text(json.dumps({"status": status}), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) not in (5, 6) or args[0] != "worker" or (len(args) == 6 and args[5] != "--subset"):
        print("usage: python -m skeptic.sandbox worker PROPOSAL INPUT OUTPUT STATUS [--subset]", file=sys.stderr)
        return 2
    worker(*args[1:5], subset=len(args) == 6)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

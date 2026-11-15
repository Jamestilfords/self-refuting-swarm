"""Check generated proposals on synthetic arrays in isolated CPU sandboxes.

No training, datasets, volumes or credentials are passed to the worker. Each
full, repeat and subset evaluation uses a fresh sandbox, just like E3.
"""
from __future__ import annotations

import math
import secrets
from contextlib import contextmanager

import numpy as np
import modal

from modal_jobs.config import Settings
from modal_jobs.sandbox import _score_in_sandbox, sandbox_image
from skeptic import api
from skeptic.sandbox import pack_input


settings = Settings.from_env()
agent_app = modal.App(f"{settings.app_name}-agents")


@agent_app.function(image=sandbox_image, cpu=0.25, memory=128, timeout=30,
                    block_network=True, restrict_modal_access=True)
def image_ready():
    """Register the sandbox image before the agent app starts."""


def validate_isolated(text: str, ctx: api.Ctx, score) -> np.ndarray:
    """The injected scorer owns isolation. Never import or exec proposal source here."""
    api.static_check(text)
    sha = api.proposal_sha(text)
    train = {k: getattr(ctx.train, k) for k in ("logits", "feats", "pooled", "labels")}
    evaluation = {k: getattr(ctx.eval, k) for k in ("logits", "feats", "pooled")}
    n, k = len(ctx.eval), ctx.train.logits.shape[1]
    full = pack_input(train, evaluation)
    scores, probs, _ = score(text, sha, full, n, k, return_probs=True)
    repeated, repeated_probs, _ = score(text, sha, full, n, k, return_probs=True)
    if not np.allclose(scores, repeated, rtol=1e-6, atol=1e-8) \
            or (probs is None) != (repeated_probs is None) \
            or (probs is not None and not np.allclose(probs, repeated_probs, rtol=1e-6, atol=1e-8)):
        raise api.Malformed("nondeterministic")
    rng = np.random.default_rng(secrets.randbits(128))
    idx = np.sort(rng.choice(n, max(2, math.ceil(api.SUBSET_FRAC * n)), replace=False))
    payload = pack_input(train, {key: value[idx] for key, value in evaluation.items()})
    try:
        subset, subset_probs, _ = score(text, sha, payload, len(idx), k, subset=True, return_probs=True)
    except api.Malformed as exc:
        if exc.reason in ("timeout", "sandbox_error"):
            raise
        raise api.Malformed("subset_dependent") from exc
    if not np.allclose(subset, scores[idx], rtol=1e-6, atol=1e-8) \
            or (probs is None) != (subset_probs is None) \
            or (probs is not None and not np.allclose(subset_probs, probs[idx], rtol=1e-6, atol=1e-8)):
        raise api.Malformed("subset_dependent")
    return scores


@contextmanager
def modal_validator(ctx: api.Ctx):
    # The CLI loads .env before importing this module. Modal functions must be
    # registered at module scope; a nested @app.function fails before app.run.
    with modal.enable_output(), agent_app.run(environment_name=settings.environment):
        def score(text, sha, payload, n, k, **kwargs):
            return _score_in_sandbox(agent_app, sandbox_image, text, sha, payload, n, k, **kwargs)
        yield lambda text: validate_isolated(text, ctx, score)

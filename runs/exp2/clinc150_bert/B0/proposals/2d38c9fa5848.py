import numpy as np
from scipy.special import digamma

FAMILY = "logit"
ARM = "edl"
RATIONALE = (
    "The expected categorical entropy under an EDL Dirichlet posterior uses both total evidence "
    "and how that evidence is distributed across intents, so it can identify ambiguous OOS inputs "
    "without being a monotone transform of energy or vacuity."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    alpha = np.asarray(ctx.h.alpha(ctx.eval.logits), dtype=float)
    alpha = np.maximum(alpha, 1e-8)
    total = alpha.sum(axis=1, keepdims=True)
    expected_entropy = digamma(total + 1.0).ravel()
    expected_entropy -= np.sum(
        (alpha / total) * digamma(alpha + 1.0),
        axis=1,
    )
    return np.nan_to_num(expected_entropy, nan=0.0, posinf=0.0, neginf=0.0)

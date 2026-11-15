import numpy as np
from scipy.special import xlogy

FAMILY = "other"
ARM = "edl"
RATIONALE = "The normalized entropy of Dirichlet concentration proportions measures whether evidential mass is diffusely spread across intents, independently of its total amount. OOS queries may produce less intent-specific concentration shapes even when vacuity or energy alone is uninformative."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    alpha = np.asarray(ctx.h.alpha(ctx.eval.logits), dtype=float)
    alpha = np.maximum(alpha, 1e-12)
    probs = alpha / np.sum(alpha, axis=1, keepdims=True)
    return -np.sum(xlogy(probs, probs), axis=1)

import numpy as np
from scipy.special import digamma, gammaln

FAMILY = "other"
ARM = "edl"
RATIONALE = "Dirichlet differential entropy uses the full concentration shape, assigning higher scores to broad and weakly concentrated evidential posteriors rather than relying only on total evidence. This can flag OOS inputs whose evidence is diffuse across intents."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    alpha = np.asarray(ctx.h.alpha(ctx.eval.logits), dtype=float)
    alpha = np.clip(alpha, 1e-8, 1e8)

    total = np.sum(alpha, axis=1)
    k = alpha.shape[1]

    log_beta = np.sum(gammaln(alpha), axis=1) - gammaln(total)
    entropy = (
        log_beta
        + (total - k) * digamma(total)
        - np.sum((alpha - 1.0) * digamma(alpha), axis=1)
    )

    return np.nan_to_num(entropy, nan=0.0, posinf=1e12, neginf=-1e12)

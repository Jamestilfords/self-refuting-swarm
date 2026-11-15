import numpy as np
from scipy.special import logsumexp

FAMILY = "other"
ARM = "ce"
RATIONALE = "A ViM-style virtual OOS logit is proportional to feature energy outside the principal in-scope subspace. Inputs with large residual semantic content receive a larger virtual-class posterior even when their ordinary logits are confident."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0071"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    mean = np.mean(tr.feats, axis=0)
    centered = tr.feats - mean
    d = centered.shape[1]
    keep = min(128, max(d - 1, 1))

    cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    basis = vecs[:, -keep:]

    train_residual = centered - (centered @ basis) @ basis.T
    eval_centered = ev.feats - mean
    eval_residual = eval_centered - (eval_centered @ basis) @ basis.T

    train_norm = np.sqrt(np.sum(train_residual * train_residual, axis=1))
    eval_norm = np.sqrt(np.sum(eval_residual * eval_residual, axis=1))

    logit_scale = np.mean(np.abs(np.max(tr.logits, axis=1)))
    residual_scale = np.mean(train_norm)
    alpha = logit_scale / max(residual_scale, 1e-12)
    alpha = float(np.clip(alpha, 1e-8, 1e6))

    virtual_logit = alpha * eval_norm
    return virtual_logit - logsumexp(
        np.concatenate([ev.logits, virtual_logit[:, None]], axis=1),
        axis=1,
    )
